import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, r2_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor, ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import seaborn as sns
 
# Set page config
st.set_page_config(page_title="ML Web App", layout="wide")
 
# Sidebar
st.sidebar.title("🧭 ML Web App Menuu")
section = st.sidebar.radio("Select Section", ["📂 Upload", "🔧 Encoding", "🧪 Scaling", "❓ Handling NULL Values", "🚨 Handling Outliers", "📊 Visualization", "🤖 Model Building"])
 
 
 
# Optional sidebar settings
st.sidebar.markdown("---")
plot_style = st.sidebar.selectbox("🎨 Seaborn Style", ["darkgrid", "whitegrid", "dark", "white", "ticks"])
sns.set_style(plot_style)
 
 
# App Title
st.title("🚀 End-to-End ML Web Application")
 
# Global dataset variable
df = None
 
# -------------------- SECTION: Upload --------------------
if section == "📂 Upload":
    st.header("📂 Upload Dataset")
 
    uploaded_file = st.file_uploader("Upload your CSV/XLSX file", type=["csv", "xlsx"])
 
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
 
            st.success("✅ File uploaded successfully!")
            st.write("### 📊 Preview of Data:")
            st.dataframe(df)
 
            # Save original uploaded data
            st.session_state['original_data'] = df
            st.session_state['encoded_data'] = None  # Reset encoded data when new file is uploaded
 
        except Exception as e:
            st.error(f"❌ Error while reading file: {e}")
 
    elif 'original_data' in st.session_state:
        st.info("ℹ️ Showing previously uploaded dataset:")
        st.dataframe(st.session_state['original_data'])
 
        # RESET button
        if st.button("🔄 Reset Uploaded Data"):
            st.session_state.pop('original_data', None)
            st.session_state.pop('encoded_data', None)
            st.success("✅ Dataset cleared. You can now upload a new file.")
 
    else:
        st.warning("⚠️ Please upload a dataset to proceed.")
 
 
 
# -------------------- SECTION: Encoding --------------------
elif section == "🔧 Encoding":
    st.header("🔠 Encoding Categorical Columns")
 
    if 'original_data' not in st.session_state or st.session_state['original_data'] is None:
        st.warning("⚠️ Please upload a dataset first in the Upload section.")
    else:
        df = st.session_state['original_data'].copy()
 
        st.write("### 📋 Current Dataset Preview (Original Data):")
        st.dataframe(df.head())
 
        # Detect categorical columns
        cat_cols = df.select_dtypes(include='object').columns.tolist()
 
        if not cat_cols:
            st.info("✅ No categorical columns found to encode.")
        else:
            st.write(f"🧾 Categorical Columns Detected: {cat_cols}")
            encoding_method = st.selectbox("Choose Encoding Method", ["Label Encoding", "One-Hot Encoding"])
 
            if st.button("🚀 Apply Encoding"):
                if encoding_method == "Label Encoding":
                    le = LabelEncoder()
                    for col in cat_cols:
                        df[col] = le.fit_transform(df[col].astype(str))
                    st.success("✅ Label Encoding applied.")
 
                elif encoding_method == "One-Hot Encoding":
                    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
                    st.success("✅ One-Hot Encoding applied.")
 
                # Save encoded version separately
                st.session_state['encoded_data'] = df
 
                st.write("### 🔄 Encoded Data Preview:")
                st.dataframe(df.head())
 
        if st.button("🔁 Reset Encoded Data"):
            st.session_state['encoded_data'] = None
            st.warning("🧹 Encoded data reset! No encoding applied currently.")
 
# -------------------- SECTION: Scaling --------------------
elif section == "🧪 Scaling":
    st.header("📐 Feature Scaling (Normalization & Standardization)")
 
    # Use encoded data if available, else original
    if 'encoded_data' in st.session_state and st.session_state['encoded_data'] is not None:
        df = st.session_state['encoded_data'].copy()
    elif 'original_data' in st.session_state and st.session_state['original_data'] is not None:
        df = st.session_state['original_data'].copy()
    else:
        st.warning("⚠️ Please upload and encode a dataset first.")
        st.stop()
 
    # Select only numeric columns
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
 
    if not num_cols:
        st.info("❌ No numerical columns available for scaling.")
    else:
        st.write("### 🧮 Numerical Columns:")
        #st.write(num_cols)
 
        scaler_choice = st.selectbox("🔍 Choose Scaler", ["MinMax Scaler", "Standard Scaler"])
        scale_cols = st.multiselect("📌 Select columns to scale", num_cols, default=num_cols)
 
        if st.button("🚀 Apply Scaling"):
            from sklearn.preprocessing import MinMaxScaler, StandardScaler
 
            scaler = MinMaxScaler() if scaler_choice == "MinMax Scaler" else StandardScaler()
            df_scaled = df.copy()
            df_scaled[scale_cols] = scaler.fit_transform(df[scale_cols])
 
            st.session_state['scaled_data'] = df_scaled
            st.success(f"✅ {scaler_choice} applied successfully!")
            st.write("### 🔍 Scaled Data Preview:")
            st.dataframe(df_scaled.head())
 
        if 'scaled_data' in st.session_state:
            if st.button("🔄 Reset Scaled Data"):
                st.session_state.pop('scaled_data', None)
                st.warning("🧹 Scaled data reset!")
 
#---------------------SECTION: Handling NULL Values -----------------
 
elif section == "❓ Handling NULL Values":
    st.header("🧼 Missing Value Handling")
 
    if 'encoded_data' not in st.session_state:
        st.warning("⚠️ Please upload a dataset first.")
    else:
        df = st.session_state['encoded_data']
        missing_info = df.isnull().sum()
        missing_info = missing_info[missing_info > 0]
 
        if missing_info.empty:
            st.success("✅ No missing values detected in the dataset.")
        else:
            st.write("### 📋 Columns with Missing Values")
            st.dataframe(missing_info.to_frame(name="Missing Count"))
 
            column_to_fix = st.selectbox("🔍 Select Column to Handle", missing_info.index.tolist())
            handling_method = st.selectbox("🛠 Choose Handling Method", [
                "Fill with Mean",
                "Fill with Median",
                "Fill with Mode",
                "Forward Fill",
                "Backward Fill",
                "Replace with Custom Value"
            ])
 
            if handling_method == "Replace with Custom Value":
                custom_value = st.text_input("🧾 Enter Custom Value", value="0")
 
            if st.button("✅ Apply"):
                try:
                    if handling_method == "Fill with Mean":
                        df[column_to_fix] = df[column_to_fix].fillna(df[column_to_fix].mean())
                        st.success(f"✅ Filled missing values in `{column_to_fix}` with **mean**.")
 
                    elif handling_method == "Fill with Median":
                        df[column_to_fix] = df[column_to_fix].fillna(df[column_to_fix].median())
                        st.success(f"✅ Filled missing values in `{column_to_fix}` with **median**.")
 
                    elif handling_method == "Fill with Mode":
                        mode_val = df[column_to_fix].mode().iloc[0]
                        df[column_to_fix] = df[column_to_fix].fillna(mode_val)
                        st.success(f"✅ Filled missing values in `{column_to_fix}` with **mode**.")
 
                    elif handling_method == "Forward Fill":
                        df[column_to_fix] = df[column_to_fix].fillna(method='ffill')
                        st.success(f"✅ Forward filled missing values in `{column_to_fix}`.")
 
                    elif handling_method == "Backward Fill":
                        df[column_to_fix] = df[column_to_fix].fillna(method='bfill')
                        st.success(f"✅ Backward filled missing values in `{column_to_fix}`.")
 
                    elif handling_method == "Replace with Custom Value":
                        df[column_to_fix] = df[column_to_fix].fillna(custom_value)
                        st.success(f"✅ Replaced missing values in `{column_to_fix}` with `{custom_value}`.")
 
                    # Update session state
                    st.session_state['encoded_data'] = df
 
                    st.write("### 🧪 Updated Dataset Preview:")
                    st.dataframe(df.head())
 
                except Exception as e:
                    st.error(f"❌ Error: {e}")
 
 
#---------------------SECTION: Outliers -----------------
elif section == "🚨 Handling Outliers":
    st.header("🚨 Outlier Detection & Capping")
 
    if 'encoded_data' not in st.session_state:
        st.warning("⚠️ Please upload a dataset first.")
    else:
        df = st.session_state['encoded_data']
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
 
        if not numeric_cols:
            st.info("ℹ️ No numeric columns found to analyze.")
        else:
            st.write("### 📦 Boxplot Visualization (Before Capping) for All Numeric Columns")
            fig, axs = plt.subplots(nrows=len(numeric_cols), ncols=1, figsize=(10, 5 * len(numeric_cols)))
            if len(numeric_cols) == 1:
                axs = [axs]  # ensure it's iterable
            for i, col in enumerate(numeric_cols):
                sns.boxplot(y=df[col], ax=axs[i])
                axs[i].set_title(f"Boxplot for {col}")
            st.pyplot(fig)
 
            total_outliers = 0
            for col in numeric_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
 
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                total_outliers += len(outliers)
 
            st.write(f"🧮 **Total Outliers detected across all numeric columns:** {total_outliers}")
 
            if st.button("🛠 Apply Capping to All"):
                for col in numeric_cols:
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    df[col] = df[col].apply(lambda x: lower_bound if x < lower_bound else upper_bound if x > upper_bound else x)
 
                st.session_state['encoded_data'] = df
                st.success("✅ Outliers for all numeric columns have been capped using IQR bounds.")
                st.write(f"📐 **Shape of dataset after capping:** {df.shape}")
 
                st.write("### 📦 Boxplot After Capping for All Numeric Columns")
                fig2, axs2 = plt.subplots(nrows=len(numeric_cols), ncols=1, figsize=(10, 5 * len(numeric_cols)))
                if len(numeric_cols) == 1:
                    axs2 = [axs2]
                for i, col in enumerate(numeric_cols):
                    sns.boxplot(y=df[col], ax=axs2[i])
                    axs2[i].set_title(f"Boxplot for {col} (After Capping)")
                st.pyplot(fig2)
 
 
 
 
# -------------------- SECTION: Visualization --------------------
elif section == "📊 Visualization":
    st.header("📊 Data Visualization")
 
    if 'encoded_data' not in st.session_state:
        st.warning("⚠️ Please upload a dataset first.")
    else:
        df = st.session_state['encoded_data']
        st.dataframe(df.head())
 
        analysis_type = st.selectbox("Select Analysis Type", ["Univariate", "Bivariate", "Multivariate"])
 
        # -------------------- UNIVARIATE --------------------
        if analysis_type == "Univariate":
            col = st.selectbox("Select Column", df.columns)
            plot_type = st.selectbox("Plot Type", [
                "Histogram", "Boxplot", "Barplot", "KDE Plot", "Violin Plot", "Countplot", "Distplot"
            ])
 
            if st.button("Generate Univariate Plot"):
                plt.figure(figsize=(8, 4))
                if plot_type == "Histogram":
                    sns.histplot(df[col], kde=False)
                elif plot_type == "Boxplot":
                    sns.boxplot(x=df[col])
                elif plot_type == "Barplot":
                    sns.barplot(x=df[col].value_counts().index, y=df[col].value_counts().values)
                elif plot_type == "KDE Plot":
                    sns.kdeplot(df[col], fill=True)
                elif plot_type == "Violin Plot":
                    sns.violinplot(x=df[col])
                elif plot_type == "Countplot":
                    sns.countplot(x=df[col])
                elif plot_type == "Distplot":
                    sns.histplot(df[col], kde=True)
                st.pyplot(plt)
 
        # -------------------- BIVARIATE --------------------
        elif analysis_type == "Bivariate":
            col1 = st.selectbox("Select X-axis Column", df.columns)
            col2 = st.selectbox("Select Y-axis Column", df.columns)
            plot_type = st.selectbox("Plot Type", [
                "Scatterplot", "Boxplot", "Barplot", "Lineplot", "Violin Plot", "Stripplot", "Swarmplot", "Jointplot", "Regplot"
            ])
 
            if st.button("Generate Bivariate Plot"):
                plt.figure(figsize=(8, 4))
                if plot_type == "Scatterplot":
                    sns.scatterplot(x=df[col1], y=df[col2])
                elif plot_type == "Boxplot":
                    sns.boxplot(x=df[col1], y=df[col2])
                elif plot_type == "Barplot":
                    sns.barplot(x=df[col1], y=df[col2])
                elif plot_type == "Lineplot":
                    sns.lineplot(x=df[col1], y=df[col2])
                elif plot_type == "Violin Plot":
                    sns.violinplot(x=df[col1], y=df[col2])
                elif plot_type == "Stripplot":
                    sns.stripplot(x=df[col1], y=df[col2])
                elif plot_type == "Swarmplot":
                    sns.swarmplot(x=df[col1], y=df[col2])
                elif plot_type == "Regplot":
                    sns.regplot(x=df[col1], y=df[col2])
                    st.pyplot(plt)
                    st.write("Regplot with regression line")
                   
                elif plot_type == "Jointplot":
                    fig = sns.jointplot(x=df[col1], y=df[col2], kind="scatter")
                    st.pyplot(fig)
                   
                st.pyplot(plt)
 
        # -------------------- MULTIVARIATE --------------------
        elif analysis_type == "Multivariate":
            plot_type = st.selectbox("Plot Type", ["Heatmap", "Pairplot", "FacetGrid", "Andrews Curves", "Parallel Coordinates"])
 
            if plot_type == "Heatmap":
                if st.button("Generate Heatmap"):
                    numeric_df = df.select_dtypes(include=['int64', 'float64'])
                    plt.figure(figsize=(10, 6))
                    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
                    st.pyplot(plt)
 
            elif plot_type == "Pairplot":
                if st.button("Generate Pairplot"):
                    numeric_df = df.select_dtypes(include=['int64', 'float64'])
                    fig = sns.pairplot(numeric_df)
                    st.pyplot(fig)
 
            elif plot_type == "FacetGrid":
                col = st.selectbox("Select Categorical Column for Facet", df.select_dtypes(include='object').columns)
                numeric_col = st.selectbox("Select Numeric Column to Plot", df.select_dtypes(include=['int64', 'float64']).columns)
                if st.button("Generate FacetGrid"):
                    g = sns.FacetGrid(df, col=numeric_col)
                    g.map(sns.histplot, numeric_col)
                    st.pyplot(g.fig)
 
            elif plot_type == "Andrews Curves":
                from pandas.plotting import andrews_curves
                # cat_col = st.selectbox("Select Class Column (Categorical)", df.select_dtypes(include='object').columns)
                numeric_col = st.selectbox("Select Numeric Column to Plot", df.select_dtypes(include=['int64', 'float64']).columns)
                if st.button("Generate Andrews Curves"):
                    fig = plt.figure(figsize=(10, 5))
                    andrews_curves(df, class_column=numeric_col)
 
 
 
# -------------------- SECTION: Model Building --------------------
 
elif section == "🤖 Model Building":
    st.header("🤖 Machine Learning Model Building")
 
    if 'encoded_data' not in st.session_state:
        st.warning("⚠️ Please upload and prepare a dataset first.")
    else:
        df = st.session_state['encoded_data']
        st.write("### 📋 Current Data Preview:")
        st.dataframe(df.head())
 
        has_target = st.checkbox("✅ I have a target/output column (Supervised Learning)", value=True)
 
        if has_target:
            st.markdown("### 🎯 Select Target Column (for Supervised Learning)")
            target_column = st.selectbox("Choose the target variable (y)", df.columns)
 
            model_type = "Classification" if df[target_column].dtype == 'object' or df[target_column].nunique() <= 20 else "Regression"
            st.info(f"🔍 Detected Learning Type: **Supervised** | Problem Type: **{model_type}**")
 
            feature_columns = [col for col in df.columns if col != target_column]
 
            model_map_cls = {
                "Logistic Regression": LogisticRegression(),
                "Random Forest Classifier": RandomForestClassifier(),
                "Decision Tree Classifier": DecisionTreeClassifier(),
                "Gradient Tree Classifier": GradientBoostingClassifier(),
                "Extra Trees Classifier": ExtraTreesClassifier(),
                "KNN Classifier": KNeighborsClassifier(),
                "SVM Classifier": SVC()
            }
 
            model_map_reg = {
                "Linear Regression": LinearRegression(),
                "Random Forest Regressor": RandomForestRegressor(),
                "Decision Tree Regressor": DecisionTreeRegressor(),
                "Gradient Boosting Regressor": GradientBoostingRegressor(),
                "Extra Trees Regressor": ExtraTreesRegressor(),
                "KNN Regressor": KNeighborsRegressor(),
                "SVM Regressor": SVR()
            }
 
            model_choice = st.selectbox("Choose a Model", list(model_map_cls.keys()) if model_type == "Classification" else list(model_map_reg.keys()))
            X = df[feature_columns]
            y = df[target_column]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
            if st.button("🚀 Train Model and Predict"):
                # Train initial model
                model = model_map_cls[model_choice] if model_type == "Classification" else model_map_reg[model_choice]
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)
 
                # Calculate initial score
                score = accuracy_score(y_test, predictions) if model_type == "Classification" else r2_score(y_test, predictions)
                metric_name = "Accuracy" if model_type == "Classification" else "R² Score"
 
                st.write(f"### ✅ Initial {metric_name}: {score:.4f}")
                st.session_state['initial_score'] = score
 
                # Define hyperparameter grid for tuning
                param_grids = {
                    "Random Forest Classifier": {"n_estimators": [50, 100, 150], "max_depth": [None, 10, 20]},
                    "Random Forest Regressor": {"n_estimators": [50, 100, 150], "max_depth": [None, 10, 20]},
                    "Logistic Regression": {"C": [0.1, 1.0, 10.0]},
                    "Linear Regression": {},  # No tuning params
                    "KNN Classifier": {"n_neighbors": [3, 5, 7]},
                    "KNN Regressor": {"n_neighbors": [3, 5, 7]},
                    "SVM Classifier": {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]},
                    "SVM Regressor": {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]},
                    "Gradient Tree Classifier": {"n_estimators": [50, 100], "learning_rate": [0.01, 0.1]},
                    "Gradient Boosting Regressor": {"n_estimators": [50, 100], "learning_rate": [0.01, 0.1]},
                }
 
                # Decide whether to tune
                if score < 0.7 and model_choice in param_grids:
                    st.warning(f"⚠️ {metric_name} is below 70%. Trying Hyperparameter Tuning to improve...")
 
                    with st.spinner("🔍 Tuning Hyperparameters..."):
                        base_model = model_map_cls[model_choice] if model_type == "Classification" else model_map_reg[model_choice]
                        param_grid = param_grids[model_choice]
 
                        # You can switch to RandomizedSearchCV if preferred
                        tuner = GridSearchCV(base_model, param_grid, cv=5, n_jobs=-1)
                        tuner.fit(X_train, y_train)
 
                        best_model = tuner.best_estimator_
                        tuned_predictions = best_model.predict(X_test)
                        tuned_score = accuracy_score(y_test, tuned_predictions) if model_type == "Classification" else r2_score(y_test, tuned_predictions)
 
                        st.session_state['trained_model'] = best_model
                        st.session_state['tuned_score'] = tuned_score
                        st.session_state['target_column'] = target_column
                        st.session_state['feature_columns'] = feature_columns
 
                        # Show result
                        st.success(f"✅ {metric_name} After Tuning: {tuned_score:.4f}")
 
                        # Compare before and after
                        st.markdown("### 📊 Before vs After Tuning Comparison")
                        comparison_df = pd.DataFrame({
                            "Metric": [metric_name],
                            "Before Tuning": [score],
                            "After Tuning": [tuned_score]
                        })
                        st.dataframe(comparison_df)
 
                else:
                    # If tuning is not needed or not applicable
                    st.success("✅ Model performance is satisfactory. No tuning needed.")
                    st.session_state['trained_model'] = model
                    st.session_state['target_column'] = target_column
                    st.session_state['feature_columns'] = feature_columns
 
 
 
        else:
            st.info("🔍 Detected Learning Type: **Unsupervised Learning**")
            st.markdown("### 🧠 Choose an Unsupervised Learning Technique")
            unsupervised_choice = st.selectbox("Select Technique", ["KMeans Clustering", "Hierarchical Clustering", "PCA (Principal Component Analysis)"])
 
            features = df.select_dtypes(include=['float64', 'int64'])
 
            if features.empty:
                st.warning("⚠️ No numerical features available for unsupervised techniques.")
            else:
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(features)
 
                if unsupervised_choice == "KMeans Clustering":
                    num_clusters = st.slider("Select Number of Clusters (K)", 2, 10, 3)
 
                    inertia_list = [KMeans(n_clusters=k, random_state=42).fit(X_scaled).inertia_ for k in range(1, 11)]
 
                    fig, ax = plt.subplots()
                    ax.plot(range(1, 11), inertia_list, marker='o')
                    ax.set_title("Elbow Method")
                    st.pyplot(fig)
 
                    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
                    labels = kmeans.fit_predict(X_scaled)
                    sil_score = silhouette_score(X_scaled, labels)
                    st.write(f"**Silhouette Score:** {sil_score:.4f}")
 
                elif unsupervised_choice == "Hierarchical Clustering":
                    num_clusters = st.slider("Select Number of Clusters", 2, 10, 3)
                    fig3, ax3 = plt.subplots(figsize=(10, 4))
                    linkage_matrix = linkage(X_scaled, method='ward')
                    dendrogram(linkage_matrix, truncate_mode='lastp', p=12, leaf_rotation=45, ax=ax3)
                    st.pyplot(fig3)
 
                    model = AgglomerativeClustering(n_clusters=num_clusters)
                    labels = model.fit_predict(X_scaled)
                    sil_score = silhouette_score(X_scaled, labels)
                    st.write(f"**Silhouette Score:** {sil_score:.4f}")
 
                elif unsupervised_choice == "PCA (Principal Component Analysis)":
                    n_components = st.slider("Number of Components", 2, min(10, X_scaled.shape[1]), 2)
                    pca = PCA(n_components=n_components)
                    components = pca.fit_transform(X_scaled)
                    cum_var = pca.explained_variance_ratio_.cumsum()
 
                    fig4, ax4 = plt.subplots()
                    ax4.plot(range(1, n_components + 1), cum_var, marker='o')
                    ax4.set_title("Cumulative Explained Variance")
                    st.pyplot(fig4)
 
                    for i in range(n_components):
                        df[f'PC{i+1}'] = components[:, i]
 
                    if n_components >= 2:
                        fig5, ax5 = plt.subplots()
                        sns.scatterplot(x=components[:, 0], y=components[:, 1], ax=ax5)
                        st.pyplot(fig5)
 
                st.success("✅ Unsupervised learning technique applied successfully.")
                st.dataframe(df.head())
 
    # ------------------ 🔮 Prediction Section ------------------
    st.subheader("🔍 Make a Prediction with Custom Input")
 
    if 'trained_model' in st.session_state and 'feature_columns' in st.session_state:
        model = st.session_state['trained_model']
        feature_columns = st.session_state['feature_columns']
        df = st.session_state['encoded_data']
        target_column = st.session_state.get('target_column')
 
        user_input = {}
        with st.form("prediction_form"):
            for col in feature_columns:
                if df[col].dtype in ['int64', 'float64']:
                    val = st.number_input(f"Enter value for {col}", value=float(df[col].mean()))
                else:
                    val = st.text_input(f"Enter value for {col}", value=str(df[col].iloc[0]))
                user_input[col] = val
 
            submitted = st.form_submit_button("🎯 Predict")
 
        if submitted:
            input_df = pd.DataFrame([user_input])
            for col in feature_columns:
                if df[col].dtype in ['int64', 'float64']:
                    input_df[col] = pd.to_numeric(input_df[col])
 
            prediction = model.predict(input_df)[0]
            st.success(f"✅ Predicted Value: **{prediction}**")