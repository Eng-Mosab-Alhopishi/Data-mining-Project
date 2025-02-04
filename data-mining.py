import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, silhouette_score, classification_report, roc_curve, auc
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier

class Logger:
    @staticmethod
    def log_step(step_name, description):
        with open("documentation.md", "a", encoding="utf-8") as doc_file:
            doc_file.write(f"## {step_name}\n{description}\n\n")

    @staticmethod
    def save_plot(fig, filename):
        if not os.path.exists("plots"):
            os.makedirs("plots")
        fig.savefig(f"plots/{filename}.png")
        Logger.log_step("Save Plot", f"Plot saved as {filename}.png")

class DataLoader:
    @staticmethod
    @st.cache_data
    def load_data(uploaded_file):
        try:
            data = pd.read_csv(uploaded_file)
            Logger.log_step("Load Data", "Data successfully loaded.")
            return data
        except Exception as e:
            Logger.log_step("Load Data", f"Error loading data: {e}")
            st.error("Failed to load data.")
            return None

class Apriori:
    @staticmethod
    def preprocess(data, selected_cols):
        apriori_data = data[selected_cols].copy()
        categorical_cols = apriori_data.select_dtypes(include=["object"]).columns
        if len(categorical_cols) > 0:
            apriori_data = pd.get_dummies(apriori_data, columns=categorical_cols, drop_first=True)
        apriori_data = apriori_data.applymap(lambda x: 1 if x > 0 else 0)
        return apriori_data

    @staticmethod
    def run(apriori_data, min_support, min_confidence):
        frequent_itemsets = apriori(apriori_data, min_support=min_support, use_colnames=True)
        if len(frequent_itemsets) > 0:
            rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
            return rules
        else:
            return None

class NaiveBayes:
    @staticmethod
    def preprocess(data):
        nb_data = data.copy()
        numerical_cols = nb_data.select_dtypes(include=[np.number]).columns
        categorical_cols = nb_data.select_dtypes(include=["object"]).columns
        
        imputer = SimpleImputer(strategy="mean")
        nb_data[numerical_cols] = imputer.fit_transform(nb_data[numerical_cols])
        
        label_encoders = {}
        for col in categorical_cols:
            label_encoders[col] = LabelEncoder()
            nb_data[col] = label_encoders[col].fit_transform(nb_data[col])
        
        return nb_data, label_encoders

    @staticmethod
    def run(X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = GaussianNB()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        acc = accuracy_score(y_test, predictions)
        cm = confusion_matrix(y_test, predictions)
        report = classification_report(y_test, predictions, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        probabilities = model.predict_proba(X_test)
        return acc, cm, report_df, probabilities, model.classes_

class DecisionTree:
    @staticmethod
    def preprocess(data):
        dt_data = data.copy()
        numerical_cols = dt_data.select_dtypes(include=[np.number]).columns
        categorical_cols = dt_data.select_dtypes(include=["object"]).columns
        
        imputer = SimpleImputer(strategy="mean")
        dt_data[numerical_cols] = imputer.fit_transform(dt_data[numerical_cols])
        
        label_encoders = {}
        for col in categorical_cols:
            label_encoders[col] = LabelEncoder()
            dt_data[col] = label_encoders[col].fit_transform(dt_data[col])
        
        return dt_data, label_encoders

    @staticmethod
    def run(X, y, max_depth):
        model = DecisionTreeClassifier(criterion="entropy", max_depth=max_depth, random_state=42)
        model.fit(X, y)
        return model

class KMeansClustering:
    @staticmethod
    def preprocess(data):
        kmeans_data = data.copy()
        numerical_cols = kmeans_data.select_dtypes(include=[np.number]).columns
        categorical_cols = kmeans_data.select_dtypes(include=["object"]).columns
        
        imputer = SimpleImputer(strategy="mean")
        kmeans_data[numerical_cols] = imputer.fit_transform(kmeans_data[numerical_cols])
        
        label_encoders = {}
        for col in categorical_cols:
            label_encoders[col] = LabelEncoder()
            kmeans_data[col] = label_encoders[col].fit_transform(kmeans_data[col])
        
        return kmeans_data, label_encoders

    @staticmethod
    def run(X, n_clusters):
        model = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = model.fit_predict(X)
        score = silhouette_score(X, clusters)
        return clusters, score

class MainApp:
    def __init__(self):
        self.logger = Logger()
        self.data_loader = DataLoader()
        self.apriori = Apriori()
        self.naive_bayes = NaiveBayes()
        self.decision_tree = DecisionTree()
        self.kmeans = KMeansClustering()

    def run(self):
        st.title("Advanced Data Mining Pipeline")
        st.write("Comprehensive data analysis and machine learning platform")
        
        # Sidebar for explanations
        st.sidebar.header("User Guide")
        st.sidebar.markdown("""
        **Visualization Legend**:
        - 🔴: Critical points to note
        - 🔵: Positive results
        - 🟢: Safe zones
        
        **Reading Tips**:
        1. Start with the main results at the top
        2. Check actionable recommendations
        3. Use filters to re-analyze data
        """)
        
        uploaded_file = st.file_uploader("Upload CSV file", type="csv")
        
        if uploaded_file:
            data = self.data_loader.load_data(uploaded_file)
            
            if data is not None:
                st.subheader("Raw Data Preview")
                st.dataframe(data.head())
                st.write(f"Dataset Shape: {data.shape}")
                
                # Apriori Section
                self.run_apriori(data)
                
                # Naive Bayes Section
                self.run_naive_bayes(data)
                
                # Decision Tree Section
                self.run_decision_tree(data)
                
                # K-Means Section
                self.run_kmeans(data)

    def run_apriori(self, data):
        st.subheader("Step 1: Association Rule Mining (Apriori)")
        st.markdown("""
        **Scientific Explanation**:
        - Confidence: Probability of consequent given antecedent
        - Lift: Strength of association (>1 means positive correlation)
        - Support: Frequency of the rule in the dataset
        """)
        
        apriori_cols = st.multiselect("Select columns for Apriori", data.columns, key="apriori_cols")
        
        if apriori_cols:
            apriori_data = self.apriori.preprocess(data, apriori_cols)
            
            st.write("#### Processed Data for Apriori")
            st.dataframe(apriori_data.head())
            
            min_support = st.slider("Minimum Support", 0.01, 1.0, 0.05, key="apriori_support")
            min_confidence = st.slider("Minimum Confidence", 0.1, 1.0, 0.5, key="apriori_confidence")
            
            if st.button("Run Apriori"):
                try:
                    rules = self.apriori.run(apriori_data, min_support, min_confidence)
                    
                    if rules is not None:
                        st.write("### Results")
                        st.dataframe(rules.sort_values("lift", ascending=False))
                        
                        # Visualization
                        fig, ax = plt.subplots()
                        rules.head(10).sort_values('lift', ascending=False).plot(kind='barh', 
                                                                                x='antecedents', 
                                                                                y='lift', 
                                                                                ax=ax)
                        ax.set_title('Top Rules by Lift Score')
                        st.pyplot(fig)
                        
                        # Business Recommendations
                        st.subheader("Actionable Recommendations")
                        best_rule = rules.iloc[0]
                        st.success(f"**Key Insight**: When customers buy **{list(best_rule['antecedents'])[0]}**, " 
                                  f"they are **{best_rule['confidence'] *100:.1f}%** likely to also buy **{list(best_rule['consequents'])[0]}** "
                                  f"(Lift = {best_rule['lift']:.2f})")
                    else:
                        st.warning("No frequent itemsets found. Try lowering the minimum support.")
                except Exception as e:
                    st.error(f"Error in Apriori: {str(e)}")
        else:
            st.warning("Please select at least one column for Apriori.")

    def run_naive_bayes(self, data):
        st.subheader("Step 2: Classification (Naive Bayes)")
        st.markdown("""
        **Evaluation Metrics**:
        - Precision: Correct predictions for each class
        - Recall: Correctly identified instances of each class
        - F1-Score: Balanced measure of precision and recall
        """)
        
        nb_data, label_encoders = self.naive_bayes.preprocess(data)
        
        st.write("#### Processed Data for Naive Bayes")
        st.dataframe(nb_data.head())
        
        target_col_nb = st.selectbox("Select Target Variable", nb_data.columns, key="nb_target")
        feature_cols_nb = st.multiselect("Select Feature Columns", nb_data.columns, key="nb_features")
        
        test_size = st.slider("Select Test Size Ratio", 0, 100, 50, key="test_size")

        if target_col_nb and feature_cols_nb:
            if st.button("Run Naive Bayes"):
                X = nb_data[feature_cols_nb]
                y = nb_data[target_col_nb]

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, 
                    test_size=test_size/100, 
                    random_state=42,
                    stratify=y  # Added for better class distribution
                )
                
                model = GaussianNB()
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)

                acc = accuracy_score(y_test, predictions)
                cm = confusion_matrix(y_test, predictions)
                report = classification_report(y_test, predictions, output_dict=True)
                report_df = pd.DataFrame(report).transpose()
                probabilities = model.predict_proba(X_test)
                
                # Store accuracy for comparison
                st.session_state.nb_acc = acc
                
                st.write("### Results")
                st.write(f"**Accuracy**: {acc:.2%}")
                
                # Confusion Matrix
                st.write("### Confusion Matrix")
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
                st.pyplot(fig)
                
                # Classification Report
                st.write("### Classification Report")
                st.dataframe(report_df)
                
                # ROC Curve
                st.write("### ROC Curve")
                
                # Handle multi-class and binary cases
                n_classes = len(model.classes_)
                
                if n_classes == 2:
                    # Binary classification
                    fpr, tpr, _ = roc_curve(y_test, probabilities[:, 1])
                    roc_auc = auc(fpr, tpr)
                    
                    fig, ax = plt.subplots()
                    ax.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
                    ax.plot([0, 1], [0, 1], 'k--')
                    ax.set_xlabel('False Positive Rate')
                    ax.set_ylabel('True Positive Rate')
                    ax.set_title('ROC Curve - Binary Classification')
                    ax.legend()
                    st.pyplot(fig)
                else:
                    # Multi-class classification
                    y_test_bin = label_binarize(y_test, classes=model.classes_)
                    
                    # Compute ROC curve and ROC area for each class
                    fpr = dict()
                    tpr = dict()
                    roc_auc = dict()
                    for i in range(n_classes):
                        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], probabilities[:, i])
                        roc_auc[i] = auc(fpr[i], tpr[i])
                    
                    # Plot all ROC curves
                    fig, ax = plt.subplots()
                    colors = plt.cm.rainbow(np.linspace(0, 1, n_classes))
                    for i, color in zip(range(n_classes), colors):
                        ax.plot(fpr[i], tpr[i], color=color,
                                label=f'Class {model.classes_[i]} (AUC = {roc_auc[i]:.2f})')
                    
                    ax.plot([0, 1], [0, 1], 'k--')
                    ax.set_xlabel('False Positive Rate')
                    ax.set_ylabel('True Positive Rate')
                    ax.set_title('ROC Curves - Multi-class')
                    ax.legend(loc="lower right")
                    st.pyplot(fig)
                
                if st.checkbox("Show Prediction Probabilities"):
                    st.write("### Prediction Probabilities")
                    st.dataframe(pd.DataFrame(probabilities, columns=model.classes_))
        else:
            st.warning("Please select a target variable and at least one feature column.")

    def run_decision_tree(self, data):
        st.subheader("Step 3: Decision Tree (ID3)")
        st.markdown("""
        **Tree Explanation**:
        - Depth: Number of decision levels
        - Node: Question/condition on a specific feature
        - Leaves: Final decision
        """)
        
        dt_data, label_encoders = self.decision_tree.preprocess(data)
        
        st.write("### Processed Data for Decision Tree")
        st.dataframe(dt_data.head())
        
        target_col_dt = st.selectbox("Select Target Variable", dt_data.columns, key="dt_target")
        feature_cols_dt = st.multiselect("Select Feature Columns", dt_data.columns, key="dt_features")
        max_depth = st.slider("Max Tree Depth", 1, 10, 3, key="tree_depth")
        
        if target_col_dt and feature_cols_dt:
            if st.button("Run Decision Tree"):
                X = dt_data[feature_cols_dt]
                y = dt_data[target_col_dt]
                model = self.decision_tree.run(X, y, max_depth)
                
                # Store accuracy for comparison
                st.session_state.dt_acc = model.score(X, y)
                
                st.write("### Decision Rules")
                fig, ax = plt.subplots(figsize=(20, 10))
                plot_tree(model, filled=True, feature_names=X.columns, class_names=[str(c) for c in y.unique()], ax=ax)
                st.pyplot(fig)
                
                st.write("### Feature Importance")
                importance = pd.Series(model.feature_importances_, index=X.columns)
                st.bar_chart(importance.sort_values(ascending=False))
                
                # Decision Path Explanation
                st.subheader("Decision Path Explanation")
                sample = X.sample(1)
                st.write("Sample Data:", sample)
                
                path = model.decision_path(sample).toarray()[0]
                features_used = [col for col, val in zip(X.columns, path) if val == 1]
                
                st.markdown("**Decision Path:**")
                for step, feature in enumerate(features_used):
                    st.write(f"Step {step+1}: Check {feature}")
        else:
            st.warning("Please select a target variable and at least one feature column.")

    def run_kmeans(self, data):
        st.subheader("Step 4: Clustering (K-Means)")
        st.markdown("""
        **Evaluation Metrics**:
        - Silhouette Score: Measure of cluster separation (-1 to 1)
        - Inertia: Sum of squared distances within clusters
        """)
        
        kmeans_data, label_encoders = self.kmeans.preprocess(data)
        
        st.write("### Processed Data for K-Means")
        st.dataframe(kmeans_data.head())
        
        feature_cols_kmeans = st.multiselect("Select Feature Columns", kmeans_data.columns, key="kmeans_features")
        n_clusters = st.slider("Number of Clusters", 2, 10, 3, key="kmeans_clusters")
        
        if feature_cols_kmeans:
            if st.button("Run K-Means"):
                X = kmeans_data[feature_cols_kmeans]
                clusters, score = self.kmeans.run(X, n_clusters)
                
                st.write("### Results")
                st.write(f"**Silhouette Score**: {score:.2f}")
                
                st.write("### Cluster Distribution")
                cluster_counts = pd.Series(clusters).value_counts().sort_index()
                st.bar_chart(cluster_counts)
                
                st.write("### Cluster Visualization")
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.scatterplot(x=X.iloc[:, 0], y=X.iloc[:, 1], hue=clusters, palette="viridis", ax=ax)
                ax.set_title("Customer Segmentation")
                st.pyplot(fig)
                
                # Cluster Profiles
                st.subheader("Cluster Characteristics")
                X['Cluster'] = clusters
                cluster_profiles = X.groupby('Cluster').mean()
                st.dataframe(cluster_profiles.style.background_gradient(cmap='Blues'))
                
                # Marketing Recommendations
                st.subheader("Marketing Recommendations")
                st.markdown("""
                - Clusters with high values in X: Targeted campaigns for premium customers
                - Clusters with low values in Y: Promotional offers to increase engagement
                - Mixed clusters: Further analysis of purchasing behavior
                """)
        else:
            st.warning("Please select at least one feature column.")

if __name__ == "__main__":
    app = MainApp()
    app.run()
