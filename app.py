from __future__ import annotations

from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


APP_DIR = Path(__file__).resolve().parent
DATA_PATH = APP_DIR / "dataset.csv"
RANDOM_STATE = 42
MAX_SEGMENTS = 8

st.set_page_config(
	page_title="Customer Spending Behavior",
	page_icon="🛍️",
	layout="wide",
	initial_sidebar_state="expanded",
)

st.markdown(
	"""
	<style>
	  .stApp { background: #f5f7fb; }
	  [data-testid="stSidebar"] { background: #101b32; }
	  [data-testid="stSidebar"] * { color: #f4f7ff; }
	  .main .block-container { max-width: 1440px; padding-top: 2rem; padding-bottom: 3rem; }
	  .hero { padding: 1.6rem 1.8rem; border-radius: 18px; color: white;
			  background: linear-gradient(115deg,#15274b 0%,#2853a5 58%,#6087df 100%);
			  margin-bottom: 1.3rem; box-shadow: 0 12px 32px rgba(30,58,115,.15); }
	  .hero h1 { margin: 0 0 .35rem 0; font-size: 2rem; }
	  .hero p { margin: 0; opacity: .86; font-size: 1rem; }
	  div[data-testid="stMetric"] { background: white; border: 1px solid #e7ebf3;
		  padding: 1rem 1.1rem; border-radius: 14px; box-shadow: 0 5px 18px rgba(26,43,77,.04); }
	  div[data-testid="stMetricLabel"] p { color: #66738c; }
	  .section-note { color: #65718a; margin-top: -.45rem; }
	  .segment-card { background: white; border: 1px solid #e7ebf3; border-radius: 14px;
		  padding: 1rem 1.1rem; margin: .45rem 0; }
	  .small-muted { color: #65718a; font-size: .92rem; }
	  .stPlotlyChart { background: white; border: 1px solid #e7ebf3; border-radius: 14px; padding: .2rem; }
	</style>
	""",
	unsafe_allow_html=True,
)


def make_demo_data(rows: int = 360) -> pd.DataFrame:
	"""Create clearly labeled example data when the bundled CSV is empty/unavailable."""
	rng = np.random.default_rng(RANDOM_STATE)
	group_sizes = [rows // 4] * 4
	for index in range(rows % 4):
		group_sizes[index] += 1
	profiles = [
		(27, 34, 0.34, 2.3, 65),
		(72, 78, 0.29, 5.9, 145),
		(78, 31, 0.31, 2.0, 120),
		(35, 80, 0.27, 5.2, 85),
	]
	parts: list[pd.DataFrame] = []
	customer_number = 1
	for size, (income_mean, spend_mean, age_sd, frequency_mean, transaction_mean) in zip(group_sizes, profiles):
		ages = np.clip(rng.normal(40, 40 * age_sd, size), 18, 75).round().astype(int)
		parts.append(
			pd.DataFrame(
				{
					"CustomerID": np.arange(customer_number, customer_number + size),
					"Age": ages,
					"Annual Income (k$)": np.clip(rng.normal(income_mean, 9, size), 12, 150).round(1),
					"Spending Score (1-100)": np.clip(rng.normal(spend_mean, 10, size), 1, 100).round(1),
					"Purchase Frequency": np.clip(rng.normal(frequency_mean, 0.8, size), 0.2, 10).round(1),
					"Average Transaction Value": np.clip(rng.normal(transaction_mean, 20, size), 10, 250).round(2),
				}
			)
		)
		customer_number += size
	return pd.concat(parts, ignore_index=True).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)


def read_csv_safely(file_obj: object, label: str) -> tuple[pd.DataFrame | None, str | None]:
	try:
		data = pd.read_csv(file_obj)
		if data.empty or len(data.columns) == 0:
			return None, f"{label} has no usable rows or columns."
		return data, None
	except pd.errors.EmptyDataError:
		return None, f"{label} is empty."
	except (UnicodeDecodeError, pd.errors.ParserError, OSError, ValueError) as exc:
		return None, f"Could not read {label}: {exc}"


def load_source(uploaded_file: object | None) -> tuple[pd.DataFrame, str, str | None]:
	if uploaded_file is not None:
		loaded, issue = read_csv_safely(uploaded_file, "Uploaded CSV")
		if loaded is not None:
			return loaded, "Uploaded dataset", None
		return make_demo_data(), "Synthetic demo dataset", issue
	if DATA_PATH.exists():
		loaded, issue = read_csv_safely(DATA_PATH, "dataset.csv")
		if loaded is not None:
			return loaded, "dataset.csv", None
		return make_demo_data(), "Synthetic demo dataset", issue
	return make_demo_data(), "Synthetic demo dataset", "dataset.csv was not found; showing clearly labeled example data."


def normalize_name(value: object) -> str:
	return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def detect_numeric_features(frame: pd.DataFrame) -> list[str]:
	id_terms = ("id", "index", "phone", "postal", "zip", "customer number", "account number")
	selected: list[str] = []
	for column in frame.columns:
		name = normalize_name(column)
		compact_name = name.replace(" ", "")
		concatenated_id = re.fullmatch(r"(?:customer|client|user|account|record|row)id", compact_name) is not None
		separated_id = re.fullmatch(r"(?:(?:customer|client|user|account|record|row) )?id", name) is not None
		if concatenated_id or separated_id or any(
			term == name or name.startswith(f"{term} ") or f" {term}" in name for term in id_terms
		):
			continue
		converted = pd.to_numeric(frame[column], errors="coerce")
		non_missing = frame[column].notna().sum()
		if non_missing == 0 or converted.notna().sum() / non_missing < 0.8:
			continue
		if converted.nunique(dropna=True) < 2:
			continue
		selected.append(str(column))
	return selected


def match_feature(features: list[str], tokens: tuple[str, ...], exclusions: tuple[str, ...] = ()) -> str | None:
	scored: list[tuple[int, str]] = []
	for feature in features:
		name = normalize_name(feature)
		if any(token in name for token in exclusions):
			continue
		score = sum(1 for token in tokens if token in name)
		if score:
			scored.append((score, feature))
	return max(scored, default=(0, None), key=lambda item: item[0])[1]


def prepare_frame(raw: pd.DataFrame) -> tuple[pd.DataFrame, list[str], int, dict[str, int]]:
	frame = raw.copy()
	frame.columns = [str(col).strip() or f"Column {i + 1}" for i, col in enumerate(frame.columns)]
	frame = frame.dropna(how="all").dropna(axis=1, how="all")
	before_duplicates = len(frame)
	frame = frame.drop_duplicates().reset_index(drop=True)
	duplicate_count = before_duplicates - len(frame)
	if frame.empty:
		return frame, [], duplicate_count, {}

	features = detect_numeric_features(frame)
	for feature in features:
		frame[feature] = pd.to_numeric(frame[feature], errors="coerce")
	missing_counts = frame.isna().sum().astype(int).to_dict()
	# Impute numeric modeling columns with medians; preserve categorical gaps as a readable label.
	for feature in features:
		median = frame[feature].median()
		frame[feature] = frame[feature].fillna(0 if pd.isna(median) else median)
	for column in frame.columns:
		if column not in features and frame[column].isna().any():
			frame[column] = frame[column].fillna("Unknown")
	return frame, features, duplicate_count, missing_counts


def choose_cluster_count(scaled: np.ndarray) -> tuple[int, list[dict[str, float | int]]]:
	rows = scaled.shape[0]
	unique_rows = len(np.unique(np.round(scaled, 10), axis=0))
	upper = min(MAX_SEGMENTS, rows - 1, unique_rows)
	scores: list[dict[str, float | int]] = []
	if upper < 2:
		return 1, scores
	best_k, best_score = 2, -1.0
	for k in range(2, upper + 1):
		model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
		labels = model.fit_predict(scaled)
		score = -1.0
		if 1 < len(np.unique(labels)) < rows:
			with warnings.catch_warnings():
				warnings.simplefilter("ignore")
				score = float(silhouette_score(scaled, labels, sample_size=min(rows, 5000), random_state=RANDOM_STATE))
		scores.append({"Clusters": k, "Silhouette score": score, "Inertia": float(model.inertia_)})
		if score > best_score:
			best_k, best_score = k, score
	return best_k, scores


def build_segment_labels(frame: pd.DataFrame, features: list[str], labels: np.ndarray) -> tuple[dict[int, str], dict[int, str], str | None, str | None]:
	spend_feature = match_feature(features, ("spend", "purchase", "transaction", "amount", "revenue", "sales", "score", "expenditure"))
	income_feature = match_feature(features, ("income", "salary", "earning", "wealth", "revenue"), exclusions=("spend", "purchase", "transaction"))
	if spend_feature is None and features:
		spend_feature = max(features, key=lambda col: float(frame[col].std() or 0))
	profiles = frame.assign(_cluster=labels).groupby("_cluster")[features].mean()
	spend_median = float(frame[spend_feature].median()) if spend_feature else 0.0
	income_median = float(frame[income_feature].median()) if income_feature else 0.0
	names: dict[int, str] = {}
	explanations: dict[int, str] = {}
	for cluster_id, profile in profiles.iterrows():
		if len(profiles) == 1:
			name = "Overall Customer Base"
		else:
			spend = float(profile[spend_feature]) if spend_feature else 0.0
			income = float(profile[income_feature]) if income_feature else 0.0
			high_spend = spend >= spend_median
			if income_feature and spend_feature:
				if high_spend and income >= income_median:
					name = "High-Income, High-Spending"
				elif high_spend:
					name = "Value-Driven Spenders"
				elif income >= income_median:
					name = "Affluent, Lower-Spending"
				else:
					name = "Budget-Conscious Customers"
			elif spend_feature:
				name = "Higher-Spending Customers" if high_spend else "Lower-Spending Customers"
			else:
				strongest = str(profile.idxmax())
				name = f"Above-Average {strongest}" if profile[strongest] >= frame[strongest].median() else f"Below-Average {strongest}"
		facts: list[str] = []
		for feature in features[:4]:
			mean_value = float(profile[feature])
			overall = float(frame[feature].mean())
			direction = "above" if mean_value >= overall else "below"
			facts.append(f"average {feature} is {mean_value:,.2f} ({direction} the dataset average of {overall:,.2f})")
		explanation = "This segment is characterized by " + "; ".join(facts) + "." if facts else "No numeric profile is available for this segment."
		names[int(cluster_id)] = name
		explanations[int(cluster_id)] = explanation
	return names, explanations, spend_feature, income_feature


def train_segmentation(frame: pd.DataFrame, features: list[str]) -> dict[str, object] | None:
	if not features or frame.empty:
		return None
	try:
		matrix = frame[features].replace([np.inf, -np.inf], np.nan)
		matrix = matrix.fillna(matrix.median()).fillna(0)
		scaler = StandardScaler()
		scaled = scaler.fit_transform(matrix)
		n_clusters, evaluation = choose_cluster_count(scaled)
		model = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
		labels = model.fit_predict(scaled)
		names, explanations, spend_feature, income_feature = build_segment_labels(frame, features, labels)
		result = frame.copy()
		result["Cluster ID"] = labels.astype(int)
		result["Segment"] = [names[int(label)] for label in labels]
		return {
			"data": result,
			"features": features,
			"scaler": scaler,
			"model": model,
			"labels": labels,
			"names": names,
			"explanations": explanations,
			"spend_feature": spend_feature,
			"income_feature": income_feature,
			"evaluation": evaluation,
		}
	except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
		st.error(f"Clustering could not be completed for this dataset: {exc}")
		return None


def show_header(title: str, subtitle: str) -> None:
	st.markdown(f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p></div>', unsafe_allow_html=True)


def render_dashboard(model_data: dict[str, object], source: str, raw_count: int) -> None:
	data = model_data["data"]
	spend_feature = model_data["spend_feature"]
	income_feature = model_data["income_feature"]
	kpis = st.columns(4)
	kpis[0].metric("Total Customers", f"{len(data):,}")
	kpis[1].metric("Average Spending", f"{data[spend_feature].mean():,.2f}" if spend_feature else "Not detected")
	kpis[2].metric("Average Income", f"{data[income_feature].mean():,.2f}" if income_feature else "Not detected")
	kpis[3].metric("Customer Segments", str(data["Cluster ID"].nunique()))
	st.caption(f"Source: {source} · {len(model_data['features'])} numerical modeling features · {raw_count:,} rows after cleaning")

	st.subheader("Customer snapshot")
	left, right = st.columns([1.15, 1])
	with left:
		counts = data["Segment"].value_counts().rename_axis("Segment").reset_index(name="Customers")
		fig = px.pie(counts, values="Customers", names="Segment", hole=0.58, title="Customer segment mix", color_discrete_sequence=px.colors.qualitative.Safe)
		fig.update_traces(textposition="inside", textinfo="percent+label")
		st.plotly_chart(fig, width="stretch")
	with right:
		numeric_summary = data[model_data["features"]].describe().T.reset_index(names="Feature")
		selected = [c for c in ("Feature", "mean", "std", "min", "max") if c in numeric_summary]
		st.dataframe(numeric_summary[selected].round(2), hide_index=True, width="stretch", height=390)


def render_overview(data: pd.DataFrame, features: list[str], duplicates: int, missing: dict[str, int], source: str) -> None:
	st.subheader("Dataset health")
	c1, c2, c3, c4 = st.columns(4)
	c1.metric("Rows", f"{len(data):,}")
	c2.metric("Columns", f"{len(data.columns):,}")
	c3.metric("Duplicates removed", f"{duplicates:,}")
	c4.metric("Numeric features detected", str(len(features)))
	st.caption(f"Loaded from {source}. Modeling features are selected automatically; customer identifier-like columns are excluded.")
	if not features:
		st.warning("No suitable numerical features were detected. Upload a CSV with at least one numeric customer or spending measure to enable clustering.")
	st.markdown("#### Detected modeling features")
	st.write(", ".join(features) if features else "None")
	with st.expander("Missing values before imputation", expanded=False):
		miss = pd.DataFrame({"Column": list(missing), "Missing values": list(missing.values())})
		miss = miss[miss["Missing values"] > 0]
		if miss.empty:
			st.success("No missing values were found in the cleaned data.")
		else:
			st.dataframe(miss, hide_index=True, width="stretch")
			st.caption("Numeric feature gaps are imputed with the median; non-numeric gaps are shown as Unknown.")
	st.markdown("#### Data preview")
	st.dataframe(data.head(100), width="stretch", hide_index=True)
	st.download_button("Download cleaned dataset", data.to_csv(index=False).encode("utf-8"), "cleaned_customer_data.csv", "text/csv")


def render_segments(model_data: dict[str, object]) -> None:
	data: pd.DataFrame = model_data["data"]
	features: list[str] = model_data["features"]
	spend_feature = model_data["spend_feature"]
	grouped = data.groupby(["Cluster ID", "Segment"], as_index=False).agg(
		Customers=("Cluster ID", "size"),
		**{f"Average {feature}": (feature, "mean") for feature in features},
	)
	if spend_feature:
		grouped = grouped.rename(columns={f"Average {spend_feature}": "Average Spending"})
	st.dataframe(grouped.round(2), hide_index=True, width="stretch")
	st.markdown("#### What the segments represent")
	for cluster_id, name in model_data["names"].items():
		count = int((data["Cluster ID"] == cluster_id).sum())
		st.markdown(
			f'<div class="segment-card"><strong>Cluster {cluster_id} · {name}</strong><br>'
			f'<span class="small-muted">{count:,} customers. {model_data["explanations"][cluster_id]}</span></div>',
			unsafe_allow_html=True,
		)


def render_analytics(model_data: dict[str, object]) -> None:
	data: pd.DataFrame = model_data["data"]
	features: list[str] = model_data["features"]
	spend_feature = model_data["spend_feature"]
	income_feature = model_data["income_feature"]
	colors = "Segment"
	first, second = st.columns(2)
	with first:
		distribution = data["Segment"].value_counts().rename_axis("Segment").reset_index(name="Customers")
		fig = px.bar(distribution, x="Segment", y="Customers", color="Segment", title="Customers by segment", color_discrete_sequence=px.colors.qualitative.Safe)
		fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Customers")
		st.plotly_chart(fig, width="stretch")
	with second:
		if spend_feature:
			fig = px.histogram(data, x=spend_feature, color=colors, marginal="box", barmode="overlay", opacity=.75, title=f"Distribution of {spend_feature}")
			st.plotly_chart(fig, width="stretch")
		else:
			st.info("A spending-like feature was not detected. Showing a customer count chart instead.")
	if income_feature and spend_feature and income_feature != spend_feature:
		fig = px.scatter(data, x=income_feature, y=spend_feature, color=colors, hover_data=["Cluster ID"], title=f"{income_feature} vs {spend_feature}", opacity=.78)
		st.plotly_chart(fig, width="stretch")
	age_feature = match_feature(features, ("age", "birth"))
	if age_feature and spend_feature and age_feature != spend_feature:
		col_a, col_b = st.columns(2)
		with col_a:
			fig = px.scatter(data, x=age_feature, y=spend_feature, color=colors, title=f"{age_feature} vs {spend_feature}", opacity=.72)
			st.plotly_chart(fig, width="stretch")
		with col_b:
			fig = px.box(data, x="Segment", y=spend_feature, color="Segment", title=f"{spend_feature} by segment", points="outliers")
			fig.update_layout(showlegend=False)
			st.plotly_chart(fig, width="stretch")
	else:
		chosen = st.selectbox("Compare a feature across segments", features, key="comparison_feature")
		fig = px.box(data, x="Segment", y=chosen, color="Segment", title=f"{chosen} by segment", points="outliers")
		fig.update_layout(showlegend=False)
		st.plotly_chart(fig, width="stretch")

	st.markdown("#### Cluster map")
	if len(features) >= 2:
		matrix = model_data["scaler"].transform(data[features])
		projected = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(matrix)
		view = pd.DataFrame({"Component 1": projected[:, 0], "Component 2": projected[:, 1], "Segment": data["Segment"], "Cluster ID": data["Cluster ID"]})
		fig = px.scatter(view, x="Component 1", y="Component 2", color="Segment", hover_data=["Cluster ID"], title="Customer clusters (PCA projection)", opacity=.8)
		st.plotly_chart(fig, width="stretch")
		st.caption("PCA projects the selected standardized features into two dimensions for visualization; clustering uses all detected features.")
	else:
		fig = px.strip(data, x=features[0], color="Segment", hover_data=["Cluster ID"], title=f"Cluster view by {features[0]}")
		st.plotly_chart(fig, width="stretch")

	st.markdown("#### Model selection")
	evaluation = model_data["evaluation"]
	if evaluation:
		st.dataframe(pd.DataFrame(evaluation).round(4), hide_index=True, width="stretch")
		fig = go.Figure(go.Scatter(x=[row["Clusters"] for row in evaluation], y=[row["Inertia"] for row in evaluation], mode="lines+markers", name="Inertia"))
		fig.update_layout(title="Elbow curve", xaxis_title="Number of clusters", yaxis_title="Inertia")
		st.plotly_chart(fig, width="stretch")
	else:
		st.info("A silhouette comparison requires at least two distinct customer profiles; a single cluster is used for this dataset.")


def render_prediction(model_data: dict[str, object]) -> None:
	features: list[str] = model_data["features"]
	if not features:
		st.warning("Prediction is unavailable because the dataset has no usable numerical features.")
		return
	st.write("Enter feature values below to assign a new customer to the closest learned segment. Inputs are initialized to the dataset median.")
	with st.form("customer_prediction_form"):
		cols = st.columns(2)
		values: dict[str, float] = {}
		for index, feature in enumerate(features):
			series = pd.to_numeric(model_data["data"][feature], errors="coerce").dropna()
			default = float(series.median()) if not series.empty else 0.0
			minimum = float(series.min()) if not series.empty else default - 1
			maximum = float(series.max()) if not series.empty else default + 1
			spread = max(maximum - minimum, abs(default) * .1, 1.0)
			with cols[index % 2]:
				values[feature] = st.number_input(
					feature,
					value=default,
					min_value=minimum - spread * 2,
					max_value=maximum + spread * 2,
					step=max(spread / 100, .1),
					format="%.3f",
					help=f"Observed data range: {minimum:,.2f}–{maximum:,.2f}",
				)
		submitted = st.form_submit_button("Predict customer segment", type="primary", width="stretch")
	if submitted:
		try:
			features_array = np.asarray([[values[col] for col in features]], dtype=float)
			scaled = model_data["scaler"].transform(pd.DataFrame(features_array, columns=features))
			cluster = int(model_data["model"].predict(scaled)[0])
			segment = model_data["names"][cluster]
			st.success(f"Predicted segment: **{segment}** (Cluster {cluster})")
			st.info(model_data["explanations"][cluster])
		except (ValueError, KeyError, FloatingPointError) as exc:
			st.error(f"Prediction could not be completed. Check the entered feature values. Details: {exc}")


def main() -> None:
	with st.sidebar:
		st.markdown("## 🛍️ SpendSense")
		st.caption("Customer intelligence dashboard")
		page = st.radio(
			"Navigation",
			["Dashboard", "Dataset Overview", "Customer Segmentation", "Visual Analytics", "Customer Prediction", "About Project"],
			label_visibility="collapsed",
		)
		st.divider()
		uploaded_file = st.file_uploader("Use a different customer CSV", type=["csv"], help="The app detects numeric features from the file automatically.")
		st.caption("Your uploaded data is processed for this app session and is not saved by the application.")

	raw, source, source_issue = load_source(uploaded_file)
	if source_issue:
		st.warning(f"{source_issue} The app is using deterministic synthetic demo data so the dashboard remains usable. Upload a customer CSV to analyze your own data.")
	if source == "Synthetic demo dataset":
		st.info("Demo mode: the displayed customer data is synthetic and does not describe real customers.")
	cleaned, features, duplicate_count, missing_counts = prepare_frame(raw)
	if cleaned.empty:
		show_header("Customer Spending Behavior", "Discover useful customer groups from behavioral data.")
		st.error("The selected data contains no usable records. Upload a non-empty CSV to continue.")
		return
	if not features:
		show_header("Customer Spending Behavior", "Discover useful customer groups from behavioral data.")
		if page == "Dataset Overview":
			render_overview(cleaned, features, duplicate_count, missing_counts, source)
		else:
			st.warning("No suitable numerical customer features were found. Visit Dataset Overview for details, or upload a CSV containing numeric spending or customer attributes.")
		return
	model_data = train_segmentation(cleaned, features)
	if model_data is None:
		return

	titles = {
		"Dashboard": ("Customer Spending Behavior", "Turn customer-level spending data into actionable segments."),
		"Dataset Overview": ("Dataset Overview", "Understand data coverage, quality, and the features used for modeling."),
		"Customer Segmentation": ("Customer Segmentation", "Explore customer groups and the behavioral patterns that distinguish them."),
		"Visual Analytics": ("Visual Analytics", "Explore customer patterns with interactive, data-adaptive charts."),
		"Customer Prediction": ("Customer Prediction", "Estimate which learned customer segment best matches a new profile."),
		"About Project": ("About the Project", "A practical, explainable workflow for unsupervised customer segmentation."),
	}
	title, subtitle = titles[page]
	show_header(title, subtitle)
	if page == "Dashboard":
		render_dashboard(model_data, source, len(cleaned))
	elif page == "Dataset Overview":
		render_overview(cleaned, features, duplicate_count, missing_counts, source)
	elif page == "Customer Segmentation":
		render_segments(model_data)
	elif page == "Visual Analytics":
		render_analytics(model_data)
	elif page == "Customer Prediction":
		render_prediction(model_data)
	else:
		st.markdown(
			"""
			**Objective**  
			Group customers using the numerical characteristics present in the selected dataset, then summarize the patterns in each group.

			**Methodology**  
			Empty rows and duplicate records are removed. Candidate numeric columns are detected automatically while identifier-like fields are excluded. Missing numeric values are median-imputed, features are standardized, and K-Means is trained. Candidate cluster counts are compared using silhouette score; inertia is provided as an elbow diagnostic. Segment labels use the detected income/spending measures and observed cluster means when available.

			**Privacy and limitations**  
			This is exploratory analysis, not a validated credit, eligibility, or other high-stakes decision system. Uploads are handled in memory by the app; hosting providers may have their own operational policies. Numeric feature selection and human-readable labels are heuristic and should be reviewed for each dataset.
			"""
		)


try:
	main()
except Exception as exc:  # Keep unexpected data/UI issues actionable without exposing a traceback.
	st.error(f"The dashboard could not complete this request: {exc}")
	st.info("Check that the input is a readable CSV with customer records and numerical features, or try another file.")
