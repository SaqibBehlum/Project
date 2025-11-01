import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, silhouette_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="ML Model Explorer", layout="wide")
st.title("🤖 Machine Learning Model Explorer")

# Sidebar Navigation
st.sidebar.header("⚙️ Controls")
learning_type = st.sidebar.selectbox("Select Learning Type", ["Supervised", "Unsupervised"])
uploaded_file = st.sidebar.file_uploader("📂 Upload Dataset (CSV)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    # Handle missing values
    df = df.dropna()

    # Encode categorical columns
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    # Feature scaling
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

    # ---- SUPERVISED LEARNING ----
    if learning_type == "Supervised":
        st.sidebar.subheader("Supervised Model Options")
        target_col = st.sidebar.selectbox("🎯 Select Target Column", df.columns)

        if target_col:
            X = df_scaled.drop(columns=[target_col])
            y = df_scaled[target_col]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            model_name = st.sidebar.selectbox("Select Algorithm", 
                                              ["Decision Tree Classifier", "Random Forest Classifier", "Support Vector Machine (SVM)"])

            if model_name == "Decision Tree Classifier":
                max_depth = st.sidebar.slider("Max Depth", 1, 20, 5)
                model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)

            elif model_name == "Random Forest Classifier":
                n_estimators = st.sidebar.slider("Number of Estimators", 10, 200, 100)
                model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)

            else:  # SVM
                c_value = st.sidebar.slider("Regularization (C)", 0.01, 10.0, 1.0)
                model = SVC(C=c_value, kernel='rbf')

            if st.sidebar.button("🚀 Train Model"):
                model.fit(X_train, y_train)
                preds = model.predict(X_test)

                acc = accuracy_score(y_test, preds)
                st.subheader("📈 Model Results")
                st.write(f"**Accuracy:** {acc:.2f}")
                st.text("Classification Report:")
                st.text(classification_report(y_test, preds))

                # Confusion Matrix
                st.subheader("🔢 Confusion Matrix")
                fig, ax = plt.subplots()
                sns.heatmap(confusion_matrix(y_test, preds), annot=True, cmap="Blues", fmt='g', ax=ax)
                st.pyplot(fig)

    # ---- UNSUPERVISED LEARNING ----
    else:
        st.sidebar.subheader("Unsupervised Model Options")
        model_name = st.sidebar.selectbox("Select Algorithm", 
                                          ["KMeans", "Agglomerative Clustering", "DBSCAN"])

        if model_name == "KMeans":
            n_clusters = st.sidebar.slider("Number of Clusters (k)", 2, 10, 3)
            model = KMeans(n_clusters=n_clusters, random_state=42)

        elif model_name == "Agglomerative Clustering":
            n_clusters = st.sidebar.slider("Number of Clusters", 2, 10, 3)
            model = AgglomerativeClustering(n_clusters=n_clusters)

        else:  # DBSCAN
            eps = st.sidebar.slider("Epsilon", 0.1, 5.0, 0.5)
            min_samples = st.sidebar.slider("Min Samples", 2, 10, 5)
            model = DBSCAN(eps=eps, min_samples=min_samples)

        if st.sidebar.button("🚀 Run Clustering"):
            cluster_labels = model.fit_predict(df_scaled)

            st.subheader("📈 Cluster Results")
            st.write(f"Number of clusters: {len(set(cluster_labels))}")

            # Visualize clusters
            if df_scaled.shape[1] >= 2:
                st.subheader("🌀 Cluster Visualization")
                fig, ax = plt.subplots()
                plt.scatter(df_scaled.iloc[:, 0], df_scaled.iloc[:, 1], c=cluster_labels, cmap='rainbow')
                plt.xlabel(df_scaled.columns[0])
                plt.ylabel(df_scaled.columns[1])
                st.pyplot(fig)

            # Evaluate clustering performance if possible
            if len(set(cluster_labels)) > 1 and -1 not in cluster_labels:
                score = silhouette_score(df_scaled, cluster_labels)
                st.write(f"Silhouette Score: **{score:.2f}**")

else:
    st.info("👆 Please upload a CSV file to begin exploration.")
