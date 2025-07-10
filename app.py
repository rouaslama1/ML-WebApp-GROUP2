import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix, mean_squared_error, r2_score
from sklearn.decomposition import PCA
from preprocess import clean_data, encode_categoricals, split_data
from ml_engine import train_model, evaluate_model

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

# Load and preview dataset
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)
    st.markdown(f"**Rows:** {df.shape[0]} &nbsp;&nbsp; **Columns:** {df.shape[1]}")

    # Exploration graphs
    st.subheader("📈 Distribution des variables numériques")
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if numeric_cols:
        selected_col = st.selectbox("Sélectionnez une variable numérique pour la distribution", numeric_cols)
        fig, ax = plt.subplots()
        sns.histplot(df[selected_col], bins=30, kde=True, ax=ax)
        ax.set_title(f"Distribution de {selected_col}")
        st.pyplot(fig)

    st.subheader("📊 Matrice de corrélation")
    if len(numeric_cols) > 1:
        fig_corr, ax_corr = plt.subplots(figsize=(8, 6))
        sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', ax=ax_corr)
        st.pyplot(fig_corr)

    # Preprocessing
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

            # Visualisation clusters PCA 2D
            st.subheader("🔎 Visualisation des clusters (PCA 2D)")
            pca = PCA(n_components=2)
            X_test_2d = pca.fit_transform(X_test)
            fig, ax = plt.subplots()
            scatter = ax.scatter(X_test_2d[:, 0], X_test_2d[:, 1], c=results["labels"], cmap='viridis')
            legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
            ax.add_artist(legend1)
            ax.set_title("Projection PCA des clusters")
            st.pyplot(fig)

        elif model_choice == "Linear Regression":
            st.subheader("Regression Evaluation")
            st.metric("MSE", f"{results['mse']:.4f}")
            st.metric("R² Score", f"{results['r2']:.4f}")

            # Scatter True vs Pred
            fig, ax = plt.subplots()
            ax.scatter(y_test, results["y_pred"])
            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
            ax.set_xlabel("True Values")
            ax.set_ylabel("Predictions")
            ax.set_title("True vs Predicted")
            st.pyplot(fig)

            # Distribution erreurs
            st.subheader("📉 Distribution des erreurs")
            errors = y_test - results["y_pred"]
            fig_err, ax_err = plt.subplots()
            sns.histplot(errors, bins=30, kde=True, ax=ax_err)
            ax_err.set_title("Distribution des erreurs de prédiction")
            st.pyplot(fig_err)

        else:
            st.success("✅ Model Trained and Evaluated!")
            st.metric("🎯 Accuracy", f"{results['accuracy']:.2%}")
            st.metric("📈 F1 Score", f"{results['f1_score']:.2%}")

            st.subheader("📉 Confusion Matrix (Heatmap)")
            fig, ax = plt.subplots()
            sns.heatmap(results["confusion_matrix"], annot=True, fmt='d', cmap="Blues", ax=ax)
            st.pyplot(fig)

            if results["y_proba"] is not None and len(np.unique(y_test)) == 2:
                y_proba = results["y_proba"]
                fpr, tpr, _ = roc_curve(y_test, y_proba)
                roc_auc = auc(fpr, tpr)
                st.subheader("📈 ROC Curve")
                fig2, ax2 = plt.subplots()
                ax2.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
                ax2.plot([0, 1], [0, 1], 'k--')
                ax2.set_xlabel('False Positive Rate')
                ax2.set_ylabel('True Positive Rate')
                ax2.set_title('ROC Curve')
                ax2.legend(loc='lower right')
                st.pyplot(fig2)

            # Importance des features si disponible
            if model_choice in ["Random Forest", "XGBoost", "Decision Tree"]:
                if hasattr(model, "feature_importances_"):
                    st.subheader("🌟 Importance des features")
                    importances = model.feature_importances_
                    features = X_train.columns
                    imp_df = pd.DataFrame({"Feature": features, "Importance": importances}).sort_values(by="Importance", ascending=False)
                    fig_imp, ax_imp = plt.subplots()
                    sns.barplot(x="Importance", y="Feature", data=imp_df, ax=ax_imp)
                    st.pyplot(fig_imp)

            elif model_choice == "KNN":
                st.info("L'importance des features n'est pas disponible pour KNN.")

            elif model_choice == "SVM":
                if params.get("kernel") == "linear":
                    st.subheader("🌟 Coefficients du modèle SVM linéaire")
                    coefs = model.coef_.flatten()
                    features = X_train.columns
                    coef_df = pd.DataFrame({"Feature": features, "Coefficient": coefs}).sort_values(by="Coefficient", ascending=False)
                    fig_coef, ax_coef = plt.subplots()
                    sns.barplot(x="Coefficient", y="Feature", data=coef_df, ax=ax_coef)
                    st.pyplot(fig_coef)
                else:
                    st.info("Visualisation des coefficients disponible uniquement pour SVM linéaire.")

else:
    st.info("👈 Please upload a CSV file to get started.")

st.markdown("---")
st.caption("Developed with Streamlit | [GitHub](https://github.com/)")
