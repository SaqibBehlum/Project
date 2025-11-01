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
import matplotlib.pyplot as plt
import seaborn as sns

# App title
st.set_page_config(page_title="Machine Learning Model Explorer", layout="wide")
st.title("🤖 Machine Learning Model Explorer")

# Sidebar navigation
st.sidebar.header("🔍 Model Settings")
learning_type = st.sidebar.radio("Select Learning Type", ("Supervised", "Unsupervised"))

# File upload
uploaded_file = st.sidebar.file_uploader("Upload CSV Dataset", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    # Preprocessing
    df = df.dropna()
    label_encoders = {}
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)
    df_scaled = pd.DataFrame(scaled_data, columns=df.columns)

    # --- SUPERVISED LEARNING ---
    if learning_type == "Supervised":
        st.sidebar.subheader("Supervised Model Options")
        target_col = st.sidebar.selectbox("Select Target Column", df.columns)
        model_choice = st.sidebar.selectbox("Select Model", [
            "Decision Tree Classifier", "Random Forest Classifier", "Support Vector Machine (SVM)"
        ])

        X = df_scaled.drop(columns=[target_col])
        y = df_scaled[target_col]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        if model_choice == "Decision Tree Classifier":
            max_depth = st.sidebar.slider("Max Depth", 1, 20, 5)
            model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)

        elif model_choice == "Random Forest Classifier":
            n_estimators = st.sidebar.slider("Number of Trees", 10, 200, 100)
            model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)

        elif model_choice == "Support Vector Machine (SVM)":
            c_val = st.sidebar.slider("C (Regularization)", 0.01, 10.0, 1.0)
            model = SVC(C=c_val)

        if st.sidebar.button("Train Model"):
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            st.success(f"✅ Model Trained Successfully! Accuracy: {acc:.2f}")

            st.subheader("📈 Confusion Matrix")
            fig, ax = plt.subplots()
            sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues", ax=ax)
            st.pyplot(fig)

            st.subheader("📋 Classification Report")
            st.text(classification_report(y_test, y_pred))

    # --- UNSUPERVISED LEARNING ---
    else:
        st.sidebar.subheader("Unsupervised Model Options")
        model_choice = st.sidebar.selectbox("Select Model", ["KMeans", "Agglomerative Clustering", "DBSCAN"])

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

        if st.sidebar.button("Run Clustering"):
            clusters = model.fit_predict(df_scaled)
            df_scaled['Cluster'] = clusters

            st.success("✅ Clustering Completed!")
            st.subheader("📊 Clustered Data Preview")
            st.dataframe(df_scaled.head())

            # Visualize clusters
            if df_scaled.shape[1] > 2:
                pca_features = df_scaled.iloc[:, :2]
            else:
                pca_features = df_scaled

            st.subheader("🎨 Cluster Visualization")
            fig, ax = plt.subplots()
            sns.scatterplot(x=pca_features.iloc[:, 0], y=pca_features.iloc[:, 1],
                            hue=clusters, palette='tab10', ax=ax)
            st.pyplot(fig)

            # Silhouette score
            try:
                score = silhouette_score(df_scaled.drop(columns=['Cluster']), clusters)
                st.info(f"Silhouette Score: {score:.2f}")
            except:
                st.warning("⚠️ Silhouette score not available for this configuration.")

else:
    st.warning("📂 Please upload a CSV file to begin.")
