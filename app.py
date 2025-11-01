import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

# ------------------------- Helper Functions -------------------------
def preprocess_dataframe(df, target_col=None, task_type='Supervised'):
    df = df.copy()
    y = None

    if task_type == 'Supervised' and target_col:
        y = df[target_col]
        X = df.drop(columns=[target_col])
    else:
        X = df

    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    # Handle missing values
    if num_cols:
        num_imputer = SimpleImputer(strategy='median')
        X[num_cols] = num_imputer.fit_transform(X[num_cols])
        scaler = StandardScaler()
        X[num_cols] = scaler.fit_transform(X[num_cols])

    if cat_cols:
        for col in cat_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))

    # Encode target column if categorical
    if y is not None and (y.dtype == 'O' or not np.issubdtype(y.dtype, np.number)):
        le_y = LabelEncoder()
        y = le_y.fit_transform(y)

    return X, y

def plot_confusion(cm):
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    return fig

def plot_cluster_scatter(X, labels, title='Cluster Plot'):
    if X.shape[1] > 2:
        pca = PCA(n_components=2)
        pts = pca.fit_transform(X)
    else:
        pts = X.values
    fig, ax = plt.subplots()
    ax.scatter(pts[:, 0], pts[:, 1], c=labels, cmap='tab10', s=50)
    ax.set_title(title)
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    return fig

# ------------------------- Streamlit UI -------------------------
st.set_page_config(page_title='Machine Learning Model Explorer', layout='wide')
st.title('🎛️ Machine Learning Model Explorer')
st.markdown('Upload a dataset, choose Supervised or Unsupervised learning, train models, and visualize results.')

st.sidebar.header('Controls')
learning_type = st.sidebar.selectbox('Select Learning Type', ['Supervised', 'Unsupervised'])
uploaded_file = st.sidebar.file_uploader('Upload CSV File', type=['csv'])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader('📊 Dataset Preview')
    st.dataframe(df.head())

    target_col = None
    if learning_type == 'Supervised':
        target_col = st.sidebar.selectbox('Select Target Column', df.columns.tolist())

    # Model Selection
    st.sidebar.markdown('---')
    model_name = st.sidebar.selectbox('Select Model',
                                      ['Decision Tree', 'Random Forest', 'Support Vector Machine'] if learning_type == 'Supervised'
                                      else ['KMeans', 'Agglomerative Clustering', 'DBSCAN'])

    # Hyperparameters
    params = {}
    st.sidebar.markdown('### Model Parameters')
    if model_name == 'Decision Tree':
        params['max_depth'] = st.sidebar.slider('Max Depth', 1, 20, 5)
        params['min_samples_split'] = st.sidebar.slider('Min Samples Split', 2, 10, 2)
    elif model_name == 'Random Forest':
        params['n_estimators'] = st.sidebar.slider('Estimators', 10, 200, 100, 10)
        params['max_depth'] = st.sidebar.slider('Max Depth', 1, 50, 10)
    elif model_name == 'Support Vector Machine':
        params['C'] = st.sidebar.slider('C (Regularization)', 0.01, 10.0, 1.0)
        params['kernel'] = st.sidebar.selectbox('Kernel', ['linear', 'rbf', 'poly'])
    elif model_name == 'KMeans':
        params['n_clusters'] = st.sidebar.slider('Number of Clusters', 2, 10, 3)
    elif model_name == 'Agglomerative Clustering':
        params['n_clusters'] = st.sidebar.slider('Number of Clusters', 2, 10, 3)
    elif model_name == 'DBSCAN':
        params['eps'] = st.sidebar.slider('Epsilon (eps)', 0.1, 5.0, 0.5)
        params['min_samples'] = st.sidebar.slider('Min Samples', 1, 20, 5)

    run = st.sidebar.button('🚀 Run Model')

    if run:
        with st.spinner('Processing data...'):
            X, y = preprocess_dataframe(df, target_col if learning_type == 'Supervised' else None, learning_type)

        if learning_type == 'Supervised':
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            if model_name == 'Decision Tree':
                model = DecisionTreeClassifier(max_depth=params['max_depth'], min_samples_split=params['min_samples_split'], random_state=42)
            elif model_name == 'Random Forest':
                model = RandomForestClassifier(n_estimators=params['n_estimators'], max_depth=params['max_depth'], random_state=42)
            elif model_name == 'Support Vector Machine':
                model = SVC(C=params['C'], kernel=params['kernel'], probability=True, random_state=42)

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)

            st.subheader('✅ Model Evaluation')
            st.metric('Accuracy', f"{acc:.4f}")
            st.text(classification_report(y_test, y_pred))
            st.pyplot(plot_confusion(confusion_matrix(y_test, y_pred)))

        else:
            if model_name == 'KMeans':
                model = KMeans(n_clusters=params['n_clusters'], random_state=42)
            elif model_name == 'Agglomerative Clustering':
                model = AgglomerativeClustering(n_clusters=params['n_clusters'])
            elif model_name == 'DBSCAN':
                model = DBSCAN(eps=params['eps'], min_samples=params['min_samples'])

            labels = model.fit_predict(X)
            st.subheader('📍 Cluster Results')
            st.write(f'Unique Clusters: {np.unique(labels).tolist()}')

            if len(set(labels)) > 1 and -1 not in set(labels):
                score = silhouette_score(X, labels)
                st.metric('Silhouette Score', f"{score:.4f}")

            st.pyplot(plot_cluster_scatter(X, labels, title=f'{model_name} Clusters'))
else:
    st.info('👈 Upload a CSV file from the sidebar to get started.')
