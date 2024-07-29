import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_circles, make_classification, make_blobs
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Function to plot decision regions
def plot_decision_regions(X, y, classifier, resolution=0.02):
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('orange', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = plt.cm.RdYlBu

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], 
                    y=X[y == cl, 1],
                    alpha=0.8, 
                    c=[colors[idx]],
                    marker=markers[idx], 
                    label=cl, 
                    edgecolor='black')
        
st.image(r"innomatcis logo.webp")

st.title("Decision Surface Visualization")
st.sidebar.header("User's Input")

# Select algorithm
algorithm = st.sidebar.selectbox("Choose the algorithm", 
                                 ("KNN", "Decision Tree", "Naive Bayes", "Random Forest", "SVC", "Logistic Regression"))

# Dataset selection
dataset = st.sidebar.selectbox("Choose the decision region dataset", 
                               ("ushape", "concerticcir", "concerticcir2", "linearsep", "outlier", "overlap", "xor", "twospirals", "random", "circles", "blobs", "moons"))

# Common parameters
n_samples = st.sidebar.slider("Number of samples", 50, 500, 100)
random_state = st.sidebar.number_input("Enter the random state", min_value=0, step=1, value=42)

# Specific parameters based on dataset
if dataset in ["ushape", "moons"]:
    noise = st.sidebar.slider("Noise level", 0.0, 0.5, 0.3)
elif dataset in ["concerticcir", "concerticcir2", "circles"]:
    noise = st.sidebar.slider("Noise level", 0.0, 0.5, 0.2)
    factor = st.sidebar.slider("Factor", 0.1, 0.9, 0.5)
    if dataset == "concerticcir2":
        factor = 0.3
elif dataset in ["linearsep", "outlier", "overlap", "random"]:
    n_redundant = st.sidebar.slider("Number of redundant features", 0, 10, 0)
    class_sep = 1.0  # Default class separation for linear separation
    if dataset == "overlap":
        class_sep = st.sidebar.slider("Class separation", 0.1, 2.0, 0.5)
elif dataset == "xor":
    noise = st.sidebar.slider("Noise level", 0.0, 0.5, 0.1)
elif dataset == "twospirals":
    noise = st.sidebar.slider("Noise level", 0.0, 0.5, 0.1)
elif dataset == "blobs":
    centers = st.sidebar.slider("Number of centers", 1, 5, 3)
    cluster_std = st.sidebar.slider("Cluster standard deviation", 0.1, 2.0, 1.0)

# Test size selection
test_size = st.sidebar.slider("Test Size", 0.10, 0.30, 0.30)

# K value, distance metric, algorithm, and weights selection for KNN
if algorithm == "KNN":
    k_value = st.sidebar.slider("Select the K value", 1, 10, 5)
    metric = st.sidebar.selectbox("Distance Metric", ("euclidean", "manhattan", "minkowski"))
    knn_algorithm = st.sidebar.selectbox("Algorithm", ("auto", "ball_tree", "kd_tree", "brute"))
    knn_weights = st.sidebar.selectbox("Weights", ("uniform", "distance"))

# Dataset and parameters
if dataset == "ushape":
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
elif dataset == "concerticcir":
    X, y = make_circles(n_samples=n_samples, noise=noise, factor=factor, random_state=random_state)
elif dataset == "concerticcir2":
    X, y = make_circles(n_samples=n_samples, noise=noise, factor=factor, random_state=random_state)
elif dataset == "linearsep":
    X, y = make_classification(n_samples=n_samples, n_features=2, n_informative=2, n_redundant=n_redundant, n_clusters_per_class=1, random_state=random_state)
elif dataset == "outlier":
    X, y = make_classification(n_samples=n_samples, n_features=2, n_informative=2, n_redundant=n_redundant, n_clusters_per_class=1, weights=[0.9, 0.1], flip_y=0, random_state=random_state)
elif dataset == "overlap":
    X, y = make_classification(n_samples=n_samples, n_features=2, n_informative=2, n_redundant=n_redundant, n_clusters_per_class=1, class_sep=class_sep, random_state=random_state)
elif dataset == "xor":
    X, y = np.random.randn(n_samples, 2), np.random.randint(0, 2, n_samples)
    y = y ^ (X[:, 0] * X[:, 1] > 0)
elif dataset == "twospirals":
    n = np.sqrt(np.random.rand(n_samples // 2, 1)) * 780 * (2 * np.pi) / 360
    d1x = -np.cos(n) * n
    d1y = np.sin(n) * n
    X1 = np.hstack((d1x, d1y))
    X2 = np.hstack((-d1x, -d1y))
    X = np.vstack((X1, X2))
    y = np.hstack((np.zeros(n_samples // 2), np.ones(n_samples // 2)))
elif dataset == "random":
    X, y = make_classification(n_samples=n_samples, n_features=2, n_informative=2, n_redundant=n_redundant, random_state=random_state)
elif dataset == "circles":
    X, y = make_circles(n_samples=n_samples, noise=noise, factor=factor, random_state=random_state)
elif dataset == "blobs":
    X, y = make_blobs(n_samples=n_samples, centers=centers, cluster_std=cluster_std, random_state=random_state)
elif dataset == "moons":
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)

# Ensure X and y have consistent lengths
if X.shape[0] != y.shape[0]:
    st.error("Generated data has inconsistent sample lengths. Please adjust the parameters.")
else:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Classifier selection
    if algorithm == "KNN":
        classifier = KNeighborsClassifier(n_neighbors=k_value, metric=metric, algorithm=knn_algorithm, weights=knn_weights)
    elif algorithm == "Decision Tree":
        classifier = DecisionTreeClassifier(max_depth=5, random_state=random_state)
    elif algorithm == "Naive Bayes":
        classifier = GaussianNB()
    elif algorithm == "Random Forest":
        classifier = RandomForestClassifier(max_depth=5, random_state=random_state)
    elif algorithm == "SVC":
        classifier = SVC()
    elif algorithm == "Logistic Regression":
        classifier = LogisticRegression()

    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
# Plot decision regions
fig, ax = plt.subplots(figsize=(5, 3))
plot_decision_regions(X_test, y_test, classifier=classifier)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title(f"Decision Surface of {dataset} Dataset\nAccuracy:{accuracy:.2f}  Precision:{precision:.2f}  Recall:{recall:.2f}  F1score:{f1:.2f}")
plt.legend(loc='upper left')
st.pyplot(fig)
