# Customer Spending Behavior

An interactive Streamlit dashboard for exploring customer data and discovering customer groups with unsupervised machine learning.

## Project objective

Identify groups of customers with similar numerical characteristics and summarize how those groups differ. The application detects relevant numeric features from the supplied CSV rather than requiring a fixed column schema. It includes a clearly marked synthetic demonstration dataset when the bundled CSV is empty or unavailable.

## Features

- Dataset preview, missing-value reporting, and duplicate removal
- Automatic detection of numerical modeling features while excluding identifier-like columns
- Median imputation for missing numeric inputs and safe handling of non-numeric columns
- K-Means clustering with standardized inputs and silhouette-based cluster-count selection
- Data-informed segment names and segment profile explanations
- Dashboard KPIs, interactive Plotly charts, PCA cluster map, and model-selection diagnostics
- New-customer prediction form using the fitted scaler and K-Means model
- CSV upload, cleaned-data preview, and download
- Helpful warnings for empty or unsuitable input data

## Machine learning methodology

1. Load `dataset.csv` relative to the application, or accept a CSV uploaded in the sidebar.
2. Remove empty rows/columns and duplicate records. Detect numeric features while filtering identifier-like fields.
3. Impute missing feature values with each feature's median and standardize the feature matrix with `StandardScaler`.
4. Evaluate candidate K-Means cluster counts from 2 through at most 8 using silhouette score. If the data does not support multiple distinct clusters, use one cluster. Inertia values are displayed as an elbow diagnostic.
5. Fit the final K-Means model and derive readable segment names from actual income/spending measures and cluster averages when such measures are identifiable.

Segment names are explanatory heuristics for exploration, not a substitute for domain review. Clusters depend on the features and customer sample supplied.

## Technology stack

- Python 3.10+
- Streamlit
- pandas and NumPy
- scikit-learn
- Plotly

## Run locally

1. Create and activate a Python virtual environment.
2. Install dependencies from `requirements.txt`.
3. Place a customer CSV at `dataset.csv` (or upload one from the app sidebar).
4. Start the dashboard with `streamlit run app.py` and open the local URL printed by Streamlit.

The committed `dataset.csv` may be empty in some distributions. When that happens, the app displays synthetic demo data and clearly labels it as such; upload a real customer CSV to analyze actual data.

## Deployment

### Streamlit Community Cloud

1. Push this project to a GitHub repository that you own or can deploy.
2. In Streamlit Community Cloud, choose **Create app**, authorize GitHub if requested, and select the repository, branch, and `app.py` as the main file.
3. Deploy. Community Cloud installs the packages listed in `requirements.txt`; keep `dataset.csv` in the repository if using a bundled dataset.
4. Verify the deployed URL, dashboard pages, file loading, and prediction form.

Do not commit credentials or sensitive customer data. Public repositories and public deployments can expose their contents and outputs. Use synthetic or appropriately anonymized data for public demos.

## Project structure

```text
Customer_Spending_Behavior/
├── app.py           # Streamlit application, data preparation, model, charts, and prediction
├── dataset.csv      # Optional bundled customer dataset
├── requirements.txt # Runtime dependencies
└── README.md        # Project documentation
```
