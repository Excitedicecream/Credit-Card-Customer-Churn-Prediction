import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import RandomOverSampler

# ---------------- Streamlit Config ---------------- #
st.set_page_config(page_title="Churn Prediction", layout="wide")
st.title("Credit Card Customer Churn Prediction")
st.write("This app analyzes feature importances using an ensemble model (Random Forest, Logistic Regression, Gradient Boosting) to predict churn.") 

# ---------------- Sidebar Navigation ---------------- #
page = st.sidebar.radio("📑 Navigate", ["📊 Data Preparation", "🔮 Prediction"])

# ---------------- Load Data ---------------- #
@st.cache_data
def load_data():
    df_raw = pd.read_csv("https://raw.githubusercontent.com/Excitedicecream/CSV-Files/refs/heads/main/BankChurners.csv")
    le = LabelEncoder()

    # Drop unwanted columns
    df = df_raw.drop([
        'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1',
        'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2'
    ], axis=1)

    # Encode target
    df['Attrition_Flag'] = le.fit_transform(df['Attrition_Flag'])

    # ---------------------------
    # Remove outliers (IQR method)
    # ---------------------------
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.drop('Attrition_Flag')
    Q1 = df[numeric_cols].quantile(0.25)
    Q3 = df[numeric_cols].quantile(0.75)
    IQR = Q3 - Q1
    df_remove_outliar = df[~((df[numeric_cols] < (Q1 - 1.5 * IQR)) | (df[numeric_cols] > (Q3 + 1.5 * IQR))).any(axis=1)]

    # Prepare features and labels
    X_raw = df_remove_outliar.drop(['Attrition_Flag', 'CLIENTNUM'], axis=1)
    y_raw = df_remove_outliar['Attrition_Flag']
    X_dummy = pd.get_dummies(X_raw)

    return df_remove_outliar, X_dummy, y_raw


df, X_dummy, y_raw = load_data()

# ---------------- Shared Preparation ---------------- #
# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_dummy, y_raw, test_size=0.2, random_state=42)

# Balance dataset
ros = RandomOverSampler(random_state=42)
X_train_balanced, y_train_balanced = ros.fit_resample(X_train, y_train)

# Feature Importances (top 7)
rf_interpret = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf_interpret.fit(X_dummy, y_raw)
importances = rf_interpret.feature_importances_
feat_imp = pd.Series(importances, index=X_dummy.columns).sort_values(ascending=False)
top7 = feat_imp.head(7).index

# Function to train tuned Random Forest
@st.cache_resource
def train_best_rf(X_train, y_train, param_dist):
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
    return rf_random.best_estimator_

# Pre-train best_rf so it’s ready for prediction page
param_dist = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}
best_rf = train_best_rf(X_train_balanced[top7], y_train_balanced, param_dist)

# ================= Page 1: Data Preparation ================= #
if page == "📊 Data Preparation":

    with st.expander("🔍 Data Preview & Cleaning", expanded=True):
        st.write("### Raw Data Sample")
        st.write(df.head())
        st.write("### Predictor Variables Sample")
        st.write(X_dummy.head())
        st.write("### Target Variable (Customer Attrition)", y_raw.value_counts().to_dict())
        st.write("NA values in each column, if any", X_dummy.isna().sum()[X_dummy.isna().sum()>0])
        st.write("Total outliers removed:", len(df) - len(df_remove_outliar))
        

    # ---------------- Cached plotting functions ---------------- #
    @st.cache_data
    def plot_2d_scatter(x_feature, y_feature, X, y):
        fig, ax = plt.subplots(figsize=(10,6))
        sns.scatterplot(data=X, x=x_feature, y=y_feature, hue=y, palette='Set1', alpha=0.7, ax=ax)
        ax.set_title(f"2D Scatter Plot of {x_feature} vs {y_feature}")
        ax.set_xlabel(x_feature)
        ax.set_ylabel(y_feature)
        ax.legend(title='Churn', labels=['No Churn', 'Churn'])
        return fig

    @st.cache_data
    def plot_3d_scatter(x_feature, y_feature, z_feature, X, y):
        fig = px.scatter_3d(
            X, x=x_feature, y=y_feature, z=z_feature,
            color=y.map({0: 'No Churn', 1: 'Churn'}),
            title=f"3D Scatter Plot of {x_feature}, {y_feature}, and {z_feature}",
            labels={'color': 'Churn'}
        )
        fig.update_traces(marker=dict(size=5))
        return fig

    with st.expander("2D Visualization of Features"):
        x_2d = st.selectbox("Select X-axis feature", X_dummy.columns, index=0)
        y_2d = st.selectbox("Select Y-axis feature", X_dummy.columns, index=1)
        fig2d = plot_2d_scatter(x_2d, y_2d, X_dummy, y_raw)
        st.pyplot(fig2d)

    with st.expander("3D Visualization of Features"):
        x_3d = st.selectbox("Select X-axis feature for 3D", X_dummy.columns, index=0, key='x3d')
        y_3d = st.selectbox("Select Y-axis feature for 3D", X_dummy.columns, index=1, key='y3d')
        z_3d = st.selectbox("Select Z-axis feature for 3D", X_dummy.columns, index=2, key='z3d')
        fig3d = plot_3d_scatter(x_3d, y_3d, z_3d, X_dummy, y_raw)
        st.plotly_chart(fig3d)

    with st.expander("⚙️ Feature Importance Analysis"):
        st.write("Top 10 Feature Importances:")
        st.write(feat_imp.head(10))

        # Plot Feature Importances
        fig, ax = plt.subplots(figsize=(10,6))
        feat_imp.head(10).plot(kind='bar', ax=ax)
        ax.set_title("Top 10 Feature Importances (Random Forest)")
        ax.set_ylabel("Importance Score")
        st.pyplot(fig)

    with st.expander("⚙️ Train/Test Split & Balancing"):
        st.write("### After Balancing")
        st.write("X_train shape:", X_train_balanced.shape)
        st.write("Balanced training set distribution:", pd.Series(y_train_balanced).value_counts().to_dict()) 

    @st.cache_data
    def evaluate_topk_features(_ensemble, feature_ranking, X_train, y_train, X_test, y_test, max_k=9):
        results = {}
        for k in range(1, max_k + 1):
            top_k_features = feature_ranking[:k]
            X_train_k = X_train.iloc[:, top_k_features]
            X_test_k = X_test.iloc[:, top_k_features]
            _ensemble.fit(X_train_k, y_train)
            y_pred_k = _ensemble.predict(X_test_k)
            acc = accuracy_score(y_test, y_pred_k)
            results[k] = acc
        return results

    with st.expander("📊 Baseline Ensemble Model Results"):
        rf = RandomForestClassifier(random_state=42)
        lr = LogisticRegression(max_iter=1000, random_state=42, solver='liblinear')
        gb = GradientBoostingClassifier(random_state=42)

        ensemble = VotingClassifier(
            estimators=[('rf', rf), ('lr', lr), ('gb', gb)],
            voting='soft'
        )
        ensemble.fit(X_train_balanced, y_train_balanced)

        importances = rf.fit(X_train_balanced, y_train_balanced).feature_importances_
        feature_ranking = np.argsort(importances)[::-1]

        results = evaluate_topk_features(
            ensemble, feature_ranking,
            X_train_balanced, y_train_balanced,
            X_test, y_test, max_k=9
        )

        for k, acc in results.items():
            st.write(f"Top {k} features test accuracy: {acc:.4f}")

        fig, ax = plt.subplots(figsize=(8,5))
        ax.plot(list(results.keys()), list(results.values()), marker="o")
        ax.set_title("Model Accuracy vs. Number of Top Features")
        ax.set_xlabel("Number of Top Features Used")
        ax.set_ylabel("Test Accuracy")
        ax.grid(True)
        st.pyplot(fig)
        st.write("Based on the plot, using the top 7 features provides a good balance between model simplicity and accuracy.")

    with st.expander("🔧 Hyperparameter Tuning for Random Forest"):
        X_train_k = X_train_balanced[top7]
        X_test_k = X_test[top7]

        y_pred_best = best_rf.predict(X_test_k)

        st.write("Tuned Random Forest Test Accuracy:", accuracy_score(y_test, y_pred_best))
        st.text("Classification Report:\n" + classification_report(
            y_test, y_pred_best, target_names=['Existing Customer','Attrited Customer']
        ))

# ================= Page 2: Prediction ================= #
elif page == "🔮 Prediction":
    st.subheader("🔮 Predict Churn with Top 7 Features")

    friendly_names = {
        "Total_Trans_Amt": "Total Transaction Amount",
        "Total_Trans_Ct": "Total Transaction Count",
        "Total_Ct_Chng_Q4_Q1": "Transaction Count Change (Q4 vs Q1)",
        "Total_Revolving_Bal": "Revolving Balance",
        "Avg_Utilization_Ratio": "Average Utilization Ratio",
        "Total_Relationship_Count": "Relationship Count",
        "Total_Amt_Chng_Q4_Q1": "Transaction Amount Change (Q4 vs Q1)"
    }

    user_input = {}
    for feature in top7:
        label = friendly_names.get(feature, feature)
        if np.issubdtype(X_train_balanced[feature].dtype, np.number):
            val = st.slider(
                label,
                float(X_train_balanced[feature].min()), 
                float(X_train_balanced[feature].max()),
                float(X_train_balanced[feature].median())
            )
        else:
            val = st.selectbox(label, X_train_balanced[feature].unique())
        user_input[feature] = val

    input_df = pd.DataFrame([user_input])

    prediction, prediction_proba = None, None
    if st.button("Predict Churn"):
        prediction = best_rf.predict(input_df)[0]
        prediction_proba = best_rf.predict_proba(input_df)[0]

    if prediction is not None:
        st.subheader("Prediction Result")
        st.write("**Attrited Customer**" if prediction == 1 else "**Existing Customer**")
        st.write(f"Confidence: {prediction_proba[prediction]:.2f}")

        st.write("### Prediction Details")
        st.write(f"Existing Customer: {prediction_proba[0]:.2f}, Attrited Customer: {prediction_proba[1]:.2f}")
