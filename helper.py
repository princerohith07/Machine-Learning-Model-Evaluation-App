import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
import streamlit as st
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LinearRegression

from keras.datasets import mnist

# Function to load classification datasets
def load_classification_dataset(dataset_name, uploaded_file=None):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        return X, y
    elif dataset_name == "Iris":
        dataset = datasets.load_iris()
        X = dataset.data
        y = dataset.target
        return X, y
    elif dataset_name == "Diabetes":
        names = ['pregnant', 'glucose', 'blood_pressure', 'skin_thickness', 'insulin', 'bmi', 'diabetes_pedigree', 'age', 'outcome']
        df = pd.read_csv('/home/rohith/Downloads/archive/diabetes.csv', names=names)
        df = df.iloc[1:]
        for i in df.columns:
            df[i] = pd.to_numeric(df[i], errors='coerce')
        X = df.drop('outcome', axis=1)
        y = df['outcome']
        return X, y
    elif dataset_name == "MNIST":
        (X, y), _ = mnist.load_data()
        X = X.reshape(X.shape[0], -1)
        X = X / 255.0
        return X, y
    else:
        st.error("Invalid dataset selection!")
        return None, None

# Function to load regression datasets
def load_regression_dataset(dataset_name, uploaded_file=None):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        return X, y
    elif dataset_name == "California Housing":
        dataset = datasets.fetch_california_housing()
        X = dataset.data
        y = dataset.target
        return X, y
    else:
        st.error("Invalid dataset selection!")
        return None, None

# Function to load clustering datasets
def load_clustering_dataset(dataset_name, uploaded_file=None):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        return X, y
    elif dataset_name == "Iris":
        dataset = datasets.load_iris()
        X = dataset.data
        y = dataset.target
        return X, y
    elif dataset_name == "Wine":
        dataset = datasets.load_wine()
        X = dataset.data
        y = dataset.target
        return X, y
    elif dataset_name == "Breast Cancer":
        dataset = datasets.load_breast_cancer()
        X = dataset.data
        y = dataset.target
        return X, y
    else:
        st.error("Invalid dataset selection!")
        return None, None

# Function to select classification model
def select_classification_model(model_name):
    if model_name == "Random Forest":
        model = RandomForestClassifier()
        params = {
            "n_estimators": [10, 50, 100],
            "max_depth": [None, 5, 10, 20],
            "min_samples_split": [2, 5, 10]
        }
    elif model_name == "SVM":
        model = SVC(probability=True)
        params = {
            "C": [0.1, 1, 10, 100],
            "gamma": [0.1, 1, 10, 100],
            "kernel": ["rbf", "linear"]
        }
    else:
        st.error("Invalid model selection!")
        return None, None
    return model, params

# Function to select regression model
def select_regression_model(model_name):
    if model_name == "Random Forest":
        model = RandomForestRegressor()
        params = {
            "n_estimators": [10, 50, 100],
            "max_depth": [None, 5, 10, 20],
            "min_samples_split": [2, 5, 10]
        }
    elif model_name == "SVM":
        model = SVR()
        params = {
            "C": [0.1, 1, 10, 100],
            "gamma": [0.1, 1, 10, 100],
            "kernel": ["rbf", "linear"]
        }
    elif model_name == "Linear Regression":
        model = LinearRegression()
        params = {}
    else:
        st.error("Invalid model selection!")
        return None, None
    return model, params

def exploratory_analyse(df):
    st.write("## Data Overview")
    st.write(df.head())
    st.write("## Data Statistics")
    st.write(df.describe())
    st.write("## Data Info")
    st.write(df.info())
    st.write("## Data Shape")
    st.write(df.shape)
    st.write("## Data Columns")
    st.write(df.columns)
    st.write("## Data Missing Values")
    st.write(df.isnull().sum())

    st.write("## Data Visualization")
    st.write("### Correlation Heatmap")
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
    st.pyplot()

    st.write("### Pairplot")
    sns.pairplot(df)
    st.pyplot()

    st.write("### Distribution of Features")
    for i in df.columns[:-1]:
        plt.figure(figsize=(10, 6))
        sns.histplot(df[i], kde=True)
        st.pyplot()
        
def find_kind_of_task(df):
    y = df.iloc[:, -1]
    if len(y.unique())/len(y) < 0.05:
        return "Classification"
    else:
        return "Regression"