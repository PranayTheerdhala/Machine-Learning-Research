import pandas as pd
import numpy as np
import joblib
import gradio as gr

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor

import warnings
warnings.filterwarnings('ignore')


# === 1. Load dataset ===
def load_dataset(path):
    """
    Load CSV into a DataFrame.
    """
    return pd.read_csv(path)


# === 2. Preprocess & feature engineering ===
def preprocess_data(df, target_col):
    """
    - Drop empties/duplicates
    - Clean Acquisition_Cost string to float
    - Parse Duration to numeric days
    - Derive CTR, CPC, CPM
    - Drop unused/post-hoc columns
    - Split features and target
    - Return train/test split & preprocessor
    """
    df = df.dropna(axis=0, how='all').drop_duplicates()
    df['Acquisition_Cost'] = (
        df['Acquisition_Cost'].astype(str)
          .str.replace(r'[\$,]', '', regex=True)
          .astype(float)
    )
    df['Duration'] = (
        df['Duration'].astype(str)
          .str.extract(r'(\d+)')
          .astype(float)
    )
    df['CTR'] = df['Clicks'] / df['Impressions']
    df['CPC'] = df['Acquisition_Cost'] / df['Clicks']
    df['CPM'] = (df['Acquisition_Cost'] / df['Impressions']) * 1000
    df = df.drop(columns=[
        'Campaign_ID', 'Date', 'ROI', 'Engagement_Score',
        'Clicks', 'Impressions'
    ], errors='ignore')
    X = df.drop(columns=[target_col])
    y = df[target_col]
    num_cols = ['Duration', 'CTR', 'CPC', 'CPM']
    cat_cols = [
        'Campaign_Type', 'Channel_Used', 'Target_Audience',
        'Customer_Segment', 'Language', 'Location'
    ]
    num_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(drop='first', handle_unknown='ignore'))
    ])
    preprocessor = ColumnTransformer([
        ('num', num_pipe, num_cols),
        ('cat', cat_pipe, cat_cols)
    ])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test, preprocessor


# === 3. Train & select best of six models ===
def train_and_select(X_train, X_test, y_train, y_test, preprocessor):
    """
    Train 6 regressors, evaluate R², and return the best pipeline.
    """
    candidates = {
        'LinearRegression': LinearRegression(),
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=0.1),
        'DecisionTree': DecisionTreeRegressor(max_depth=5),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=50),
        'XGBoost': XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3)
    }
    best_name, best_score, pipelines = None, -np.inf, {}
    for name, model in candidates.items():
        pipe = Pipeline([('prep', preprocessor), ('model', model)])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        score = r2_score(y_test, preds)
        print(f"{name}: R²={score:.4f}")
        pipelines[name] = pipe
        if score > best_score:
            best_score, best_name = score, name
    print(f"\n🏆 Best model selected: {best_name} (R²={best_score:.4f})")
    return pipelines[best_name]


if __name__ == '__main__':
    data_path = r"C:\Users\mythi\OneDrive\Desktop\Final Project\marketing_campaign_dataset.csv"
    df = load_dataset(data_path)
    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(df, 'Conversion_Rate')
    best_pipeline = train_and_select(X_train, X_test, y_train, y_test, preprocessor)
    joblib.dump(best_pipeline, 'best_campaign_pipeline.pkl')

    dropdowns = {
        k: df[k].dropna().unique().tolist()
        for k in [
            'Campaign_Type','Channel_Used','Target_Audience',
            'Customer_Segment','Language','Location'
        ]
    }

    # === 4. Gradio UI with explanation ===
    def predict_with_explanation(
        Campaign_Type, Channel_Used, Target_Audience,
        Customer_Segment, Language, Location,
        Duration, Impressions, Clicks, Acquisition_Cost
    ):
        # derive CTR, CPC, CPM on the fly
        ctr = Clicks / Impressions
        cpc = Acquisition_Cost / Clicks
        cpm = (Acquisition_Cost / Impressions) * 1000

        # assemble input
        row = {
            'Duration': Duration,
            'CTR': ctr, 'CPC': cpc, 'CPM': cpm,
            'Campaign_Type': Campaign_Type,
            'Channel_Used': Channel_Used,
            'Target_Audience': Target_Audience,
            'Customer_Segment': Customer_Segment,
            'Language': Language,
            'Location': Location
        }
        X_new = pd.DataFrame([row])
        # predict
        pred_frac = best_pipeline.predict(X_new)[0]
        pct = pred_frac * 100
        est_conversions = Clicks * pred_frac

        # explanation text
        explanation = (
            f"**Predicted Conversion Rate:** {pct:.2f}%  \n"
            f"- Out of every 100 clicks, ~{pct:.1f} conversions expected.  \n"
            f"- With {Clicks} clicks, approx {est_conversions:.0f} conversions.  \n"
            f"- Use this metric to benchmark channels, optimize messaging, and allocate budget."
        )
        return pct, explanation

    iface = gr.Interface(
        fn=predict_with_explanation,
        inputs=[
            gr.Dropdown(dropdowns['Campaign_Type'], label='Campaign Type'),
            gr.Dropdown(dropdowns['Channel_Used'], label='Channel Used'),
            gr.Dropdown(dropdowns['Target_Audience'], label='Target Audience'),
            gr.Dropdown(dropdowns['Customer_Segment'], label='Customer Segment'),
            gr.Dropdown(dropdowns['Language'], label='Language'),
            gr.Dropdown(dropdowns['Location'], label='Location'),
            gr.Number(label='Duration (days)'),
            gr.Number(label='Impressions'),
            gr.Number(label='Clicks'),
            gr.Number(label='Acquisition Cost')
        ],
        outputs=[
            gr.Number(label='Conversion Rate (%)'),
            gr.Markdown(label='Explanation')
        ],
        title="Lean Conversion Predictor",
        description=(
            "Enter core campaign inputs. Behind the scenes we derive:\n"
            "- CTR = Clicks / Impressions\n"
            "- CPC = Cost / Clicks\n"
            "- CPM = Cost per 1,000 impressions\n\n"
            "The model predicts conversion rate and explains its meaning."
        )
    )

    # Launch in browser, remains until closed
    iface.launch(inbrowser=True)