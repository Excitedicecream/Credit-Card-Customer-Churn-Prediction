import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import RandomOverSampler

# ---------------- Streamlit Config ---------------- #
st.set_page_config(page_title="Churn Prediction", layout="wide")
st.title("💳 Credit Card Customer Churn Prediction")
st.write("This app analyzes key features using an ensemble model to predict customer churn.")

# ---------------- Sidebar Navigation ---------------- #
st.sidebar.header("📑 Navigation")
page = st.sidebar.radio("", ["📊 Data Preparation", "🔮 Prediction"])

st.sidebar.markdown("---")
st.sidebar.header("👤 About the Creator")
st.sidebar.markdown(
    """
**Jonathan Wong Tze Syuen**  
📚 Data Science  

🔗 [Connect on LinkedIn](https://www.linkedin.com/in/jonathan-wong-2b9b39233/)

🔗 [Connect on Github](https://github.com/Excitedicecream)
"""
)
st.sidebar.markdown("---")

# ---------------- Load Data ---------------- #
@st.cache_data
def load_data():
    df_raw = pd.read_csv("https://raw.githubusercontent.com/Excitedicecream/CSV-Files/refs/heads/main/BankChurners.csv")
    le = LabelEncoder()

    df = df_raw.drop([
        'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1',
        'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2'
    ], axis=1)

    df['Attrition_Flag'] = le.fit_transform(df['Attrition_Flag'])

    # Remove outliers (IQR method)
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.drop('Attrition_Flag')
    Q1, Q3 = df[numeric_cols].quantile(0.25), df[numeric_cols].quantile(0.75)
    IQR = Q3 - Q1
    df_no_outlier = df[~((df[numeric_cols] < (Q1 - 1.5 * IQR)) | (df[numeric_cols] > (Q3 + 1.5 * IQR))).any(axis=1)]

    X = df_no_outlier.drop(['Attrition_Flag', 'CLIENTNUM'], axis=1)
    y = df_no_outlier['Attrition_Flag']
    X_dummy = pd.get_dummies(X)

    return df_raw, df_no_outlier, X_dummy, y

df_raw, df_clean, X_dummy, y_raw = load_data()

# ---------------- Shared Preparation ---------------- #
X_train, X_test, y_train, y_test = train_test_split(X_dummy, y_raw, test_size=0.2, random_state=42)

ros = RandomOverSampler(random_state=42)
X_train_balanced, y_train_balanced = ros.fit_resample(X_train, y_train)

rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf.fit(X_dummy, y_raw)
feat_imp = pd.Series(rf.feature_importances_, index=X_dummy.columns).sort_values(ascending=False)
top8 = feat_imp.head(8).index

# ---------------- Hyperparameter Tuning ---------------- #
@st.cache_resource
def train_best_rf(X_train, y_train):
    param_dist = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }
    rf_base = RandomForestClassifier(random_state=42)
    rf_random = RandomizedSearchCV(
        estimator=rf_base,
        param_distributions=param_dist,
        n_iter=20,
        cv=3,
        verbose=0,
        random_state=42,
        n_jobs=-1
    )
    rf_random.fit(X_train, y_train)
    return rf_random.best_estimator_, rf_random.best_params_

best_rf, best_params = train_best_rf(X_train_balanced[top8], y_train_balanced)

# ================= Page 1: Data Preparation ================= #
if page == "📊 Data Preparation":

    with st.expander("🔍 Data Preview & Cleaning", expanded=True):
        st.write("### Raw Data Sample")
        st.dataframe(df_raw.head())
        st.write("**Total rows:**", len(df_raw))
        st.write("### Processed Data Sample")
        st.dataframe(X_dummy.head())
        st.write("### Target Variable Distribution")
        st.write(y_raw.value_counts().to_dict())
        st.write("**Outliers Removed:**", len(df_raw) - len(df_clean))

    with st.expander("⚙️ Feature Importance Analysis"):
        st.write("Top 10 Important Features:")
        st.dataframe(feat_imp.head(10))

        fig, ax = plt.subplots(figsize=(10, 6))
        feat_imp.head(10).plot(kind='bar', ax=ax, color='skyblue')
        ax.set_title("Top 10 Feature Importances (Random Forest)")
        ax.set_ylabel("Importance Score")
        st.pyplot(fig)

    with st.expander("🔧 Hyperparameter Tuning for Random Forest"):
        X_test_k = X_test[top8]
        y_pred_best = best_rf.predict(X_test_k)

        st.write("**Best Parameters Found:**", best_params)
        st.write("**Test Accuracy:**", round(accuracy_score(y_test, y_pred_best), 3))
        st.text("Classification Report:\n" + classification_report(
            y_test, y_pred_best, target_names=['Existing Customer', 'Attrited Customer']
        ))

    st.markdown("---")
    st.subheader("💡 Key Feature Insights")
    st.markdown("""
    - **Total_Trans_Amt** ↑ → Less churn (loyal high spenders)  
    - **Total_Trans_Ct** ↑ → Less churn (frequent transactions = engagement)  
    - **Total_Ct_Chng_Q4_Q1** ↓ → More churn (drop in activity = warning sign)  
    - **Total_Revolving_Bal** ↑ → More churn (financial stress or dissatisfaction)  
    - **Avg_Utilization_Ratio** ↑ → More churn (heavy credit use = stress)  
    - **Total_Relationship_Count** ↑ → Less churn (diverse relationship = loyalty)  
    - **Total_Amt_Chng_Q4_Q1** ↓ → More churn (spending drop = disengagement)  
    - **Credit_Limit** ↑ → Less churn (premium users stay longer)
    """)

# ================= Page 2: Prediction ================= #
elif page == "🔮 Prediction":
    st.subheader("🔮 Predict Churn with Top 8 Features")

    friendly_names = {
        "Total_Trans_Amt": "Total Transaction Amount",
        "Total_Trans_Ct": "Total Transaction Count",
        "Total_Ct_Chng_Q4_Q1": "Transaction Count Change (Q4 vs Q1)",
        "Total_Revolving_Bal": "Revolving Balance",
        "Avg_Utilization_Ratio": "Average Utilization Ratio",
        "Total_Relationship_Count": "Relationship Count",
        "Total_Amt_Chng_Q4_Q1": "Transaction Amount Change (Q4 vs Q1)",
        "Credit_Limit": "Credit Limit"
    }

    user_input = {}
    for feature in top8:
        label = friendly_names.get(feature, feature)
        val = st.slider(
            label,
            float(X_train_balanced[feature].min()),
            float(X_train_balanced[feature].max()),
            float(X_train_balanced[feature].median())
        )
        user_input[feature] = val

    input_df = pd.DataFrame([user_input])

    if st.button("Predict Churn"):
        prediction = best_rf.predict(input_df)[0]
        prediction_proba = best_rf.predict_proba(input_df)[0]

        st.subheader("🧭 Prediction Result")
        st.write("**Attrited Customer**" if prediction == 1 else "**Existing Customer**")
        st.write(f"Confidence: {prediction_proba[prediction]:.2f}")
        st.write(f"Existing Customer: {prediction_proba[0]:.2f} | Attrited Customer: {prediction_proba[1]:.2f}")

