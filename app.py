import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, silhouette_score
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="Machine Learning Model Explorer", layout="wide")
st.title("🤖 Machine Learning Model Explorer")
st.markdown("This app lets you explore **Supervised** and **Unsupervised** learning interactively.")

# -------------------------------
# SIDEBAR SETUP
# -------------------------------
st.sidebar.title("⚙️ Controls")
learning_type = st.sidebar.radio("Learning Type", ("Supervised", "Unsupervised"))
st.sidebar.markdown("---")

uploaded_file = st.sidebar.file_uploader("📂 Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    # -------------------------------
    # DATA PREPROCESSING
    # -------------------------------
    st.sidebar.markdown("### 🧹 Preprocessing Options")
    df = df.dropna()
    label_encoders = {}

    # Encode categorical data
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

    # -------------------------------
    # SUPERVISED LEARNING SECTION
    # -------------------------------
    if learning_type == "Supervised":
        st.sidebar.markdown("### 🎯 Supervised Learning")
        target_col = st.sidebar.selectbox("Select Target Column", df.columns)

        model_choice = st.sidebar.selectbox(
            "Select Model",
            ["Decision Tree Classifier", "Random Forest Classifier", "Support Vector Machine (SVM)"]
        )

        # Split data
        X = df_scaled.drop(columns=[target_col])
        y = df[target_col]  # Use original (unscaled) target

        # Ensure target is categorical for classification
        if y.nunique() > 20:
            st.warning("⚠️ Target has too many unique values; this may not be classification data.")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Model selection
        if model_choice == "Decision Tree Classifier":
            max_depth = st.sidebar.slider("Max Depth", 1, 20, 5)
            model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)

        elif model_choice == "Random Forest Classifier":
            n_estimators = st.sidebar.slider("Number of Trees", 10, 200, 100)
            model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)

        elif model_choice == "Support Vector Machine (SVM)":
            c_val = st.sidebar.slider("C (Regularization)", 0.01, 10.0, 1.0)
            kernel = st.sidebar.selectbox("Kernel", ["linear", "rbf", "poly"])
            model = SVC(C=c_val, kernel=kernel)

        st.sidebar.markdown("---")
        if st.sidebar.button("🚀 Train Model"):
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                acc = accuracy_score(y_test, y_pred)
                st.success(f"✅ Model trained successfully! Accuracy: **{acc:.2f}**")

                # Confusion Matrix
                st.subheader("📉 Confusion Matrix")
                fig, ax = plt.subplots()
                sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap="Blues", fmt="d", ax=ax)
                st.pyplot(fig)

                # Classification Report
                st.subheader("📋 Classification Report")
                st.text(classification_report(y_test, y_pred))
            except Exception as e:
                st.error(f"❌ Error during training: {e}")

    # -------------------------------
    # UNSUPERVISED LEARNING SECTION
    # -------------------------------
    elif learning_type == "Unsupervised":
        st.sidebar.markdown("### 🧠 Unsupervised Learning")
        model_choice = st.sidebar.selectbox(
            "Select Model",
            ["KMeans", "Agglomerative Clustering", "DBSCAN"]
        )

        if model_choice == "KMeans":
            k = st.sidebar.slider("Number of Clusters (k)", 2, 10, 3)
            model = KMeans(n_clusters=k, random_state=42)

        elif model_choice == "Agglomerative Clustering":
            n_clusters = st.sidebar.slider("Number of Clusters", 2, 10, 3)
            model = AgglomerativeClustering(n_clusters=n_clusters)

        elif model_choice == "DBSCAN":
            eps = st.sidebar.slider("Epsilon (eps)", 0.1, 5.0, 0.5)
            min_samples = st.sidebar.slider("Min Samples", 2, 20, 5)
            model = DBSCAN(eps=eps, min_samples=min_samples)

        st.sidebar.markdown("---")
        if st.sidebar.button("🌀 Run Clustering"):
            try:
                clusters = model.fit_predict(df_scaled)
                df_scaled["Cluster"] = clusters
                st.success("✅ Clustering Completed!")

                # Cluster Visualization
                st.subheader("🎨 Cluster Visualization")
                fig, ax = plt.subplots()
                sns.scatterplot(
                    x=df_scaled.iloc[:, 0], y=df_scaled.iloc[:, 1],
                    hue="Cluster", palette="tab10", data=df_scaled, ax=ax
                )
                st.pyplot(fig)

                # Silhouette Score
                if len(set(clusters)) > 1 and -1 not in clusters:
                    score = silhouette_score(df_scaled.drop(columns=["Cluster"]), clusters)
                    st.info(f"Silhouette Score: **{score:.2f}**")
                else:
                    st.warning("⚠️ Silhouette score not available (only one cluster detected).")

                st.subheader("📊 Clustered Data Preview")
                st.dataframe(df_scaled.head())

            except Exception as e:
                st.error(f"❌ Error during clustering: {e}")

else:
    st.info("👈 Upload a CSV file to get started.")
