import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc, mean_squared_error, r2_score
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from helper import load_classification_dataset, load_regression_dataset, load_clustering_dataset, select_classification_model, select_regression_model, exploratory_analyse, find_kind_of_task
import time

st.set_option('deprecation.showPyplotGlobalUse', False)

# Main function
def main():
    st.sidebar.title("Machine Learning Model Evaluation")
    
    # Sidebar - Page selection
    page = st.sidebar.selectbox("Select Page", ["Home", "Classification", "Regression", "Clustering"])
    
    if page == "Home":
        st.title("Welcome to Machine Learning Model Evaluation App")
        st.write("""
        ## Select a category from the sidebar to get started:
        - Classification
        - Regression
        - Clustering
        """)
    
        st.title("Upload Your Own Dataset")
        st.write("Upload your dataset as a CSV file. The file should have features in columns and the target variable as the last column.")
        
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            st.write("### File Preview")
            df = pd.read_csv(uploaded_file)
            st.write(df.head())
            explore_data = st.sidebar.checkbox("Explore Data")
            if explore_data:
                exploratory_analyse(df)

            task = find_kind_of_task(df)
            if task == "Classification":
                st.sidebar.subheader("Select Model")
                model_name = st.sidebar.selectbox("Select Model", ["Random Forest", "SVM"])
                
                # Load classification dataset
                X, y = df.iloc[:, :-1], df.iloc[:, -1]
                
                # Select classification model
                model, params = select_classification_model(model_name)
                if model is None:
                    return
                
                # Train/test split ratio
                test_size = st.sidebar.slider("Test Size", 0.1, 0.5, 0.2, 0.05)
                
                # Train model
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
                
                # Hyperparameter tuning
                st.sidebar.subheader("Hyperparameters (Optional)")
                hyperparameters = {}
                for param, values in params.items():
                    hyperparameters[param] = st.sidebar.selectbox(param, values)
                if hyperparameters:
                    model.set_params(**hyperparameters)
                
                model.fit(X_train, y_train)
                
                # Evaluate model
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                
                # Display accuracy
                st.write(f"Accuracy: {accuracy:.2f}")
                
                # Display classification report
                st.subheader("Classification Report")
                report = classification_report(y_test, y_pred)
                st.write(report)
                
                # Plot ROC curve
                if model_name in ["Random Forest", "SVM"] and len(np.unique(y)) == 2:  # Only for binary classification
                    st.subheader("ROC Curve")
                    fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
                    roc_auc = auc(fpr, tpr)
                    
                    plt.figure()
                    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
                    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                    plt.xlim([0.0, 1.0])
                    plt.ylim([0.0, 1.05])
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.title('Receiver Operating Characteristic')
                    plt.legend(loc="lower right")
                    st.pyplot(plt)
                
                # Feature Importance Visualization
                st.subheader("Feature Importance")
                if model_name == "Random Forest" and hasattr(model, 'feature_importances_'):
                    feature_importance_df = pd.DataFrame({
                        'Feature': [f'Feature {i}' for i in range(X.shape[1])],
                        'Importance': model.feature_importances_
                    })
                    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
                    plt.figure(figsize=(10, 6))
                    plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
                    plt.xlabel('Importance')
                    plt.title('Feature Importance')
                    plt.gca().invert_yaxis()  # Invert y-axis for better visualization
                    st.pyplot(plt)
                else:
                    st.warning("Feature importances are not available for this model.")
                    
            elif task == "Regression":
                st.sidebar.subheader("Select Model")
                model_name = st.sidebar.selectbox("Select Model", ["Random Forest", "SVM", "Linear Regression"])
                
                # Load regression dataset
                X, y = df.iloc[:, :-1], df.iloc[:, -1]
                
                # Select regression model
                model, params = select_regression_model(model_name)
                if model is None:
                    return
                
                # Train/test split ratio
                test_size = st.sidebar.slider("Test Size", 0.1, 0.5, 0.2, 0.05)
                
                # Train model
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
                
                # Hyperparameter tuning
                st.sidebar.subheader("Hyperparameters (Optional)")
                hyperparameters = {}
                for param, values in params.items():
                    hyperparameters[param] = st.sidebar.selectbox(param, values)
                if hyperparameters:
                    model.set_params(**hyperparameters)
                
                model.fit(X_train, y_train)
                
                # Evaluate model
                y_pred = model.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                # Display metrics
                st.write(f"Mean Squared Error: {mse:.2f}")
                st.write(f"R2 Score: {r2:.2f}")
                
                # Plot predictions vs actual values
                st.subheader("Predictions vs Actual Values")
                plt.figure(figsize=(10, 6))
                plt.scatter(y_test, y_pred)
                plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
                plt.xlabel('Actual')
                plt.ylabel('Predicted')
                plt.title('Predictions vs Actual Values')
                st.pyplot(plt)
                
            elif task == "Clustering":
                st.sidebar.subheader("Select Model")
                model_name = st.sidebar.selectbox("Select Model", ["KMeans", "DBSCAN"])
                
                # Load clustering dataset
                X, y = load_clustering_dataset(None, uploaded_file)
                
                # Select clustering model
                if model_name == "KMeans":
                    n_clusters = st.sidebar.slider("Number of Clusters", 2, 10, 3, 1)
                    model = KMeans(n_clusters=n_clusters, random_state=42)
                    labels = model.fit_predict(X)
                    centroids = model.cluster_centers_
                elif model_name == "DBSCAN":
                    eps = st.sidebar.slider("Epsilon (eps)", 0.1, 10.0, 0.5, 0.1)
                    min_samples = st.sidebar.slider("Min Samples", 1, 10, 5, 1)
                    model = DBSCAN(eps=eps, min_samples=min_samples)
                    labels = model.fit_predict(X)
                    centroids = None
                else:
                    st.error("Invalid model selection!")
                    return
                
                # Display clustering results
                st.write("### Clustering Results")
                st.write("#### Cluster Labels")
                st.write(pd.Series(labels).value_counts())
                
                if centroids is not None:
                    st.write("#### Cluster Centroids")
                    st.write(pd.DataFrame(centroids, columns=[f'Feature {i}' for i in range(centroids.shape[1])]))
                
                # Elbow method for KMeans
                if model_name == "KMeans":
                    st.write("### Elbow Method")
                    inertias = []
                    for i in range(1, 11):
                        kmeans = KMeans(n_clusters=i, random_state=42)
                        kmeans.fit(X)
                        inertias.append(kmeans.inertia_)
                    plt.figure()
                    st.set_option('deprecation.showPyplotGlobalUse', False)
                    plt.plot(range(1, 11), inertias, marker='o')
                    plt.xlabel('Number of Clusters')
                    plt.ylabel('Inertia')
                    plt.title('Elbow Method')
                    st.pyplot()
                
                # Visualize clusters using PCA for dimensionality reduction
                st.write("### Clustering Visualization (PCA)")
                pca = PCA(n_components=2)
                pca_result = pca.fit_transform(X)
                df_pca = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2'])
                df_pca['Cluster'] = labels
                st.set_option('deprecation.showPyplotGlobalUse', False) 
                sns.scatterplot(x='PC1', y='PC2', hue='Cluster', palette='viridis', data=df_pca)
                st.pyplot()

    elif page == "Classification":
        st.title("Classification Models")
        
        # Sidebar - Dataset selection for classification
        st.sidebar.subheader("Select Dataset")
        dataset_name = st.sidebar.selectbox("Select Dataset", ["Iris", "Diabetes", "MNIST"])
        
        # Load classification dataset
        X, y = load_classification_dataset(dataset_name, None)
        
        if X is None or y is None:
            return
        
        # Sidebar - Model selection for classification
        st.sidebar.subheader("Select Model")
        model_name = st.sidebar.selectbox("Select Model", ["Random Forest", "SVM"])
        
        # Select classification model
        model, params = select_classification_model(model_name)
        if model is None:
            return
        
        # Train/test split ratio
        test_size = st.sidebar.slider("Test Size", 0.1, 0.5, 0.2, 0.05)
        
        # Train model
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        # Hyperparameter tuning
        st.sidebar.subheader("Hyperparameters (Optional)")
        hyperparameters = {}
        for param, values in params.items():
            hyperparameters[param] = st.sidebar.selectbox(param, values)
        if hyperparameters:
            model.set_params(**hyperparameters)
        
        model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Display accuracy
        st.write(f"Accuracy: {accuracy:.2f}")
        
        # Display classification report
        st.subheader("Classification Report")
        report = classification_report(y_test, y_pred)
        st.write(report)
        
        # Plot ROC curve
        if model_name in ["Random Forest", "SVM"] and len(np.unique(y)) == 2:  # Only for binary classification
            st.subheader("ROC Curve")
            fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
            roc_auc = auc(fpr, tpr)
            
            plt.figure()
            plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic')
            plt.legend(loc="lower right")
            st.pyplot(plt)
        
        # Feature Importance Visualization
        st.subheader("Feature Importance")
        if model_name == "Random Forest" and hasattr(model, 'feature_importances_'):
            feature_importance_df = pd.DataFrame({
                'Feature': [f'Feature {i}' for i in range(X.shape[1])],
                'Importance': model.feature_importances_
            })
            feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
            plt.figure(figsize=(10, 6))
            plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
            plt.xlabel('Importance')
            plt.title('Feature Importance')
            plt.gca().invert_yaxis()  # Invert y-axis for better visualization
            st.pyplot(plt)
        else:
            st.warning("Feature importances are not available for this model.")

        # Data Exploration Section
        # for MNIST dataset display the sample images and according data exploration
        st.sidebar.subheader("Data Exploration")
        explore_data = st.sidebar.checkbox("Explore Data")
        if explore_data:
            if dataset_name == "MNIST":
                st.subheader("Data Exploration Section")
                st.write("### Displaying Sample Images")
                fig, axes = plt.subplots(3, 3, figsize=(10, 10))
                for i, ax in enumerate(axes.flat):
                    ax.imshow(X[i].reshape(28, 28), cmap='gray')
                    ax.axis('off')
                    ax.set_title(f"Digit: {y[i]}")
                st.pyplot(fig)
            else:
                exploratory_analyse(pd.DataFrame(X, columns=[f'Feature {i}' for i in range(X.shape[1])]))   

    elif page == "Regression":
        st.title("Regression Models")
        
        # Sidebar - Dataset selection for regression
        st.sidebar.subheader("Select Dataset")
        dataset_name = st.sidebar.selectbox("Select Dataset", ["California Housing"])
        
        # Load regression dataset
        X, y = load_regression_dataset(dataset_name, None)
        
        if X is None or y is None:
            return
        
        # Sidebar - Model selection for regression
        st.sidebar.subheader("Select Model")
        model_name = st.sidebar.selectbox("Select Model", ["Random Forest", "SVM", "Linear Regression"])
        
        # Select regression model
        model, params = select_regression_model(model_name)
        if model is None:
            return
        
        # Train/test split ratio
        test_size = st.sidebar.slider("Test Size", 0.1, 0.5, 0.2, 0.05)
        
        # Train model
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        # Hyperparameter tuning
        st.sidebar.subheader("Hyperparameters (Optional)")
        hyperparameters = {}
        for param, values in params.items():
            hyperparameters[param] = st.sidebar.selectbox(param, values)
        if hyperparameters:
            model.set_params(**hyperparameters)
        
        model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Display metrics
        st.write(f"Mean Squared Error: {mse:.2f}")
        st.write(f"R2 Score: {r2:.2f}")
        
        # Plot predictions vs actual values
        st.subheader("Predictions vs Actual Values")
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title('Predictions vs Actual Values')
        st.pyplot(plt)

        explore_data = st.sidebar.checkbox("Explore Data")
        if explore_data:
            exploratory_analyse(pd.DataFrame(X, columns=[f'Feature {i}' for i in range(X.shape[1])]))


    elif page == "Clustering":
        st.title("Clustering Models")
        
        # Sidebar - Dataset selection for clustering
        st.sidebar.subheader("Select Dataset")
        dataset_name = st.sidebar.selectbox("Select Dataset", ["Iris", "Wine", "Breast Cancer"])
        
        # Load clustering dataset
        X, y = load_clustering_dataset(dataset_name, None)
        
        if X is None or y is None:
            return
        
        # Sidebar - Model selection for clustering
        st.sidebar.subheader("Select Model")
        model_name = st.sidebar.selectbox("Select Model", ["KMeans", "DBSCAN"])
        
        # Select clustering model
        if model_name == "KMeans":
            n_clusters = st.sidebar.slider("Number of Clusters", 2, 10, 3, 1)
            model = KMeans(n_clusters=n_clusters, random_state=42)
            labels = model.fit_predict(X)
            centroids = model.cluster_centers_
        elif model_name == "DBSCAN":
            eps = st.sidebar.slider("Epsilon (eps)", 0.1, 10.0, 0.5, 0.1)
            min_samples = st.sidebar.slider("Min Samples", 1, 10, 5, 1)
            model = DBSCAN(eps=eps, min_samples=min_samples)
            labels = model.fit_predict(X)
            centroids = None
        else:
            st.error("Invalid model selection!")
            return
        
        # Display clustering results
        st.write("### Clustering Results")
        st.write("#### Cluster Labels")
        st.write(pd.Series(labels).value_counts())
        
        if centroids is not None:
            st.write("#### Cluster Centroids")
            st.write(pd.DataFrame(centroids, columns=[f'Feature {i}' for i in range(centroids.shape[1])]))
        
        # Elbow method for KMeans
        if model_name == "KMeans":
            st.write("### Elbow Method")
            inertias = []
            for i in range(1, 11):
                kmeans = KMeans(n_clusters=i, random_state=42)
                kmeans.fit(X)
                inertias.append(kmeans.inertia_)
            plt.figure()
            st.set_option('deprecation.showPyplotGlobalUse', False)
            plt.plot(range(1, 11), inertias, marker='o')
            plt.xlabel('Number of Clusters')
            plt.ylabel('Inertia')
            plt.title('Elbow Method')
            st.pyplot()
        
        # Visualize clusters using PCA for dimensionality reduction
        st.write("### Clustering Visualization (PCA)")
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(X)
        df_pca = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2'])
        df_pca['Cluster'] = labels
        st.set_option('deprecation.showPyplotGlobalUse', False) 
        sns.scatterplot(x='PC1', y='PC2', hue='Cluster', palette='viridis', data=df_pca)
        st.pyplot()

if __name__ == "__main__":
    main()
