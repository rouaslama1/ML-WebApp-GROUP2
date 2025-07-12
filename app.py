import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.metrics import roc_curve, auc, confusion_matrix, mean_squared_error, r2_score
from sklearn.decomposition import PCA
from preprocess import clean_data, encode_categoricals, split_data
from ml_engine import train_model, evaluate_model

# Streamlit page config and style
st.set_page_config(
    page_title="ML Model Selector",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.main {background-color: #f8fafc;}
.stButton>button {background-color: #2563eb; color: white;}
.stSidebar {background-color: #f1f5f9;}
</style>
""", unsafe_allow_html=True)

st.title("🤖 Machine Learning Model Selector")
st.markdown("<span style='font-size:18px;'>Upload dataset, select model, tune hyperparams, and visualize predictions.</span>", unsafe_allow_html=True)

# Sidebar upload and model selection
st.sidebar.header("1️⃣ Upload & Options")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

st.sidebar.header("2️⃣ Model Selection")
model_choice = st.sidebar.selectbox(
    "Choose a Machine Learning Model",
    ["Random Forest", "SVM", "XGBoost", "KNN", "Decision Tree", "Linear Regression", "KMeans"],
    index=0
)

# Sidebar hyperparameter selection
st.sidebar.header("3️⃣ Hyperparameters")
params = {}
if model_choice == "Random Forest":
    params["n_estimators"] = st.sidebar.slider("Number of Trees (n_estimators)", 10, 300, 100, step=10)
    params["max_depth"] = st.sidebar.slider("Max Depth", 1, 20, 5)
elif model_choice == "SVM":
    params["C"] = st.sidebar.slider("Regularization (C)", 0.01, 10.0, 1.0)
    params["kernel"] = st.sidebar.selectbox("Kernel", ["linear", "rbf", "poly"], index=0)
elif model_choice == "XGBoost":
    params["max_depth"] = st.sidebar.slider("Max Depth", 1, 10, 3)
    params["learning_rate"] = st.sidebar.slider("Learning Rate", 0.01, 0.5, 0.1)
elif model_choice == "KNN":
    params["n_neighbors"] = st.sidebar.slider("Number of Neighbors (k)", 1, 20, 5)
    params["weights"] = st.sidebar.selectbox("Weights", ["uniform", "distance"])
elif model_choice == "Decision Tree":
    params["max_depth"] = st.sidebar.slider("Max Depth", 1, 20, 5)
    params["criterion"] = st.sidebar.selectbox("Criterion", ["gini", "entropy"])
elif model_choice == "Linear Regression":
    st.sidebar.markdown("_No hyperparameters to tune for Linear Regression_")
elif model_choice == "KMeans":
    params["n_clusters"] = st.sidebar.slider("Number of Clusters", 2, 10, 3)
    params["init"] = st.sidebar.selectbox("Initialization Method", ["k-means++", "random"])
    params["max_iter"] = st.sidebar.slider("Max Iterations", 100, 500, 300)

# Main app: load and visualize dataset
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)
    st.markdown(f"**Rows:** {df.shape[0]} &nbsp;&nbsp; **Columns:** {df.shape[1]}")

    # 📈 Plot 1: Distribution of a numerical variable using Plotly histogram
    st.subheader("📈 Distribution of Numerical Variables")
    st.markdown("This chart shows how values are spread for the selected numerical column. It helps detect skewness or outliers.")
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if numeric_cols:
        selected_col = st.selectbox("Select a numerical variable for distribution", numeric_cols)
        fig = px.histogram(df, x=selected_col, nbins=30, marginal="box", 
                           title=f"Distribution of {selected_col}", opacity=0.75)
        fig.update_layout(bargap=0.1)
        st.plotly_chart(fig, use_container_width=True)

    # 📊 Plot 2: Correlation Matrix using Plotly heatmap
    st.subheader("📊 Correlation Matrix")
    st.markdown("Displays the relationships between numeric features. High positive or negative values indicate strong correlation.")
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr()
        fig_corr = px.imshow(corr_matrix, text_auto=True, color_continuous_scale='RdBu_r', title="Correlation Heatmap")
        fig_corr.update_layout(margin=dict(l=50, r=50, t=50, b=50))
        st.plotly_chart(fig_corr, use_container_width=True)


    # Clean and preprocess data
    df_clean = clean_data(df)
    df_encoded, _ = encode_categoricals(df_clean)

    st.markdown("---")
    st.subheader("⚙️ Model Configuration")
    st.write(f"**Selected Model:** {model_choice}")
    st.json(params)

    target_column = st.selectbox("🎯 Select Target Column", df_encoded.columns)

    run = st.button("🚀 Run Model", use_container_width=True)
    if run and target_column:
        X_train, X_test, y_train, y_test = split_data(df_encoded, target_column)
        model = train_model(model_choice, X_train, y_train, params)
        results = evaluate_model(model, X_test, y_test)

        if model_choice == "KMeans":
            st.subheader("Cluster Assignments for Test Data")
            st.write(results["labels"])

            # 📌 Plot 3: PCA 2D cluster visualization for KMeans
            st.subheader("🔎 Visualization of Clusters (PCA 2D)")
            st.markdown("This 2D plot shows how KMeans clusters the data after dimensionality reduction with PCA.")

            pca = PCA(n_components=2)
            X_test_2d = pca.fit_transform(X_test)
            pca_df = pd.DataFrame(X_test_2d, columns=["PC1", "PC2"])
            pca_df["Cluster"] = results["labels"]
            fig_pca = px.scatter(pca_df, x="PC1", y="PC2", color="Cluster", title="PCA Projection of Clusters", color_continuous_scale='Viridis')
            st.plotly_chart(fig_pca, use_container_width=True)

        elif model_choice == "Linear Regression":
            st.subheader("Regression Evaluation")
            st.metric("MSE", f"{results['mse']:.4f}")
            st.metric("R² Score", f"{results['r2']:.4f}")

            # 📌 Plot 4: True vs Predicted values (scatter with diagonal)
            st.markdown("This scatter plot compares actual target values with model predictions. Points near the diagonal are better predictions.")
            fig = px.scatter(x=y_test, y=results["y_pred"], 
                             labels={'x': 'True Values', 'y': 'Predicted Values'},
                             title="True vs Predicted")
            fig.add_shape(type='line', x0=y_test.min(), y0=y_test.min(), x1=y_test.max(), y1=y_test.max(),
                          line=dict(color='black', dash='dash'))
            st.plotly_chart(fig, use_container_width=True)

            # 📌 Plot 5: Distribution of errors
            st.subheader("📉 Error Distribution")
            st.markdown("Shows the spread of prediction errors. A centered, narrow distribution means better accuracy.")

            errors = y_test - results["y_pred"]
            fig_err = px.histogram(errors, nbins=30, marginal="box", title="Distribution of Prediction Errors")
            st.plotly_chart(fig_err, use_container_width=True)

        else:
            st.success("✅ Model Trained and Evaluated!")
            st.metric("🎯 Accuracy", f"{results['accuracy']:.2%}")
            st.metric("📈 F1 Score", f"{results['f1_score']:.2%}")

            # 📌 Plot 6: Confusion matrix using Plotly heatmap
            st.subheader("📉 Confusion Matrix (Heatmap)")
            st.markdown("Shows how often predicted labels match the true labels. Ideal predictions lie on the diagonal.")
            conf_matrix = results["confusion_matrix"]
            fig_cm = px.imshow(conf_matrix, text_auto=True, color_continuous_scale="Blues", title="Confusion Matrix")
            st.plotly_chart(fig_cm, use_container_width=True)

# 📌 Plot 7: ROC Curve
            st.subheader("📈 ROC Curve")
            st.markdown("Visualizes the trade-off between true and false positive rates. AUC closer to 1 means better performance.")
            try:
                if results["y_proba"] is not None:
                    unique_classes = np.unique(y_test)
                    if len(unique_classes) == 2:
                        fpr, tpr, _ = roc_curve(y_test, results["y_proba"])
                        roc_auc = auc(fpr, tpr)

                        fig_roc = px.area(
                            x=fpr, y=tpr,
                            title=f"ROC Curve (AUC = {roc_auc:.2f})",
                            labels=dict(x="False Positive Rate", y="True Positive Rate"),
                        )
                        fig_roc.add_shape(
                            type='line',
                            line=dict(dash='dash'),
                            x0=0, y0=0, x1=1, y1=1
                        )
                        st.plotly_chart(fig_roc, use_container_width=True)
                    else:
                        st.info("ROC Curve is only available for binary classification (2 classes).")
                else:
                    st.info("Probability predictions (`predict_proba`) not available for this model.")
            except Exception as e:
                st.error(f"An error occurred while plotting the ROC Curve: {e}")

            # 📌 Plot 8: Feature importance (tree models)
            if model_choice in ["Random Forest", "XGBoost", "Decision Tree"]:
                if hasattr(model, "feature_importances_"):
                    st.subheader("🌟 Feature Importance")
                    st.markdown("Highlights which features had the most impact on model decisions. Higher bars = more importance.")

                    importances = model.feature_importances_
                    features = X_train.columns
                    imp_df = pd.DataFrame({"Feature": features, "Importance": importances}).sort_values(by="Importance", ascending=False)
                    fig_imp = px.bar(imp_df, x="Importance", y="Feature", orientation="h", title="Feature Importance", color="Importance", color_continuous_scale='Blues')
                    st.plotly_chart(fig_imp, use_container_width=True)

            elif model_choice == "KNN":
                st.info("Feature importance is not available for KNN.")

            # 📌 Plot 9: SVM coefficients (only for linear kernel)
                st.subheader("🌟 Coefficients of the Linear SVM Model")
                st.markdown("Shows the weight of each feature in decision-making. Positive or negative values affect the classification.")

            elif model_choice == "SVM":
                if params.get("kernel") == "linear":
                    st.subheader("🌟 Coefficients of the Linear SVM Model")
                    coefs = model.coef_.flatten()
                    features = X_train.columns
                    coef_df = pd.DataFrame({"Feature": features, "Coefficient": coefs}).sort_values(by="Coefficient", ascending=False)
                    fig_coef = px.bar(coef_df, x="Coefficient", y="Feature", orientation="h", title="SVM Coefficients", color="Coefficient", color_continuous_scale='RdBu')
                    st.plotly_chart(fig_coef, use_container_width=True)
                else:
                    st.info("Visualization of coefficients is only available for linear SVM.")

else:
    st.info("👈 Please upload a CSV file to get started.")

st.markdown("---")
st.caption("Developed with Streamlit | [GitHub](https://github.com/)")
