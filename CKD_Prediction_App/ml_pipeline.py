import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
import pickle
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import os

print("--- Starting ML Pipeline: Data Loading, EDA, Cleaning, Training ---")

# Loading the Dataset
dataset_path = 'dataset/chronickidneydisease.csv'
try:
    df = pd.read_csv(dataset_path)
    print(f"\nDataset '{dataset_path}' loaded successfully.")
    print(f"Initial dataset shape: {df.shape}")
except FileNotFoundError:
    print(f"ERROR: Dataset '{dataset_path}' not found.")
    print("Please ensure you have downloaded 'chronickidneydisease.csv' and placed it in the 'CKD_Prediction_App/dataset/' folder.")
    exit()
except Exception as e:
    print(f"An error occurred while reading the dataset: {e}")
    exit()

# data preprocessing
df.replace('?', np.nan, inplace=True)
print("Replaced '?' with NaN across the dataset.")

if 'id' in df.columns:
    df.drop('id', axis=1, inplace=True)
    print("Column 'id' dropped.")
else:
    print("Column 'id' not found (already removed or not present).")

df.columns = df.columns.str.strip().str.lower()
print("Columns stripped and lowercased.")

new_column_names = {
    'age': 'age', 'bp': 'blood_pressure', 'sg': 'specific_gravity',
    'al': 'albumin', 'su': 'sugar', 'rbc': 'red_blood_cells',
    'pc': 'pus_cell', 'pcc': 'pus_cell_clumps', 'ba': 'bacteria',
    'bgr': 'blood_glucose_random', 'bu': 'blood_urea',
    'sc': 'serum_creatinine', 'sod': 'sodium', 'pot': 'potassium',
    'hemo': 'hemoglobin', 'pcv': 'packed_cell_volume',
    'wc': 'white_blood_cell_count', 'rc': 'red_blood_cell_count',
    'htn': 'hypertension', 'dm': 'diabetes_mellitus',
    'cad': 'coronary_artery_disease', 'appet': 'appetite',
    'pe': 'pedal_edema', 'ane': 'anemia', 'classification': 'class' 
}
df.rename(columns={k: v for k, v in new_column_names.items() if k in df.columns and k != v}, inplace=True)
print("Columns renamed based on standard mapping for readability and app compatibility.")



print("\n--- Cleaned Data Head (first 5 rows) ---")
print(df.head())

print("\n--- Data Information (non-null counts and initial data types) ---")
df.info()

print("\n--- Descriptive Statistics for Numerical Columns (Initial) ---")
print(df.describe())

print("\n--- Descriptive Statistics for Categorical Columns (Initial) ---")
print(df.describe(include='object'))

print(f"\nDataset Shape: {df.shape} (rows, columns)")
print(f"Column Names: {df.columns.tolist()}")

# Map target 'class' to numerical 0 and 1
df['class'] = df['class'].astype(str).str.strip().str.lower().replace({'ckd': 1, 'notckd': 0})
print("\nTarget 'class' mapped to numerical (0: not ckd, 1: ckd).")

# Identify numerical and categorical columns for further processing
target_col = 'class'
numerical_features = df.select_dtypes(include=np.number).columns.tolist()
numerical_features = [col for col in numerical_features if col != target_col] # Ensure target is not in features
categorical_features = df.select_dtypes(include='object').columns.tolist()

print("\n--- Identified Numerical Features (for analysis and imputation) ---")
print(numerical_features)
print("\n--- Identified Categorical Features (for analysis and imputation) ---")
print(categorical_features)
print(f"\n--- Target Variable: {target_col} ---")


# --- 3. Missing Value Analysis ---
print("\n--- Missing Values Count per Column (After initial '?' handling) ---")
missing_values_count = df.isnull().sum()
missing_values_count = missing_values_count[missing_values_count > 0].sort_values(ascending=False)
print(missing_values_count)

print("\n--- Missing Values Percentage per Column (After initial '?' handling) ---")
missing_percentage = (df.isnull().sum() / len(df)) * 100
missing_percentage = missing_percentage[missing_percentage > 0].sort_values(ascending=False)
print(missing_percentage)

print("\n--- Visualizing Missing Value Pattern (Matrix) ---")
msno.matrix(df, figsize=(15, 7), color=(0.2, 0.4, 0.6))
plt.title('Missing Value Matrix', fontsize=20)
plt.show()


# --- 4. Duplicate Data Check ---
duplicate_rows_count = df.duplicated().sum()
print(f"\nNumber of duplicate rows found: {duplicate_rows_count}")
if duplicate_rows_count > 0:
    print("Sample of Duplicate Rows:")
    print(df[df.duplicated()].head())
    df.drop_duplicates(inplace=True)
    print(f"Duplicate rows dropped. New dataset shape: {df.shape}")
else:
    print("No duplicate rows found.")


# --- 5. Univariate Analysis ---
print("\n--- Univariate Analysis (Numerical Features Histograms) ---")
plt.figure(figsize=(20, 15))
for i, col in enumerate(numerical_features):
    plt.subplot(4, 4, i + 1)
    sns.histplot(df[col].dropna(), kde=True, bins=30, color='skyblue')
    plt.title(f'Distribution of {col}', fontsize=12)
    plt.xlabel(col)
    plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

print("\n--- Univariate Analysis (Categorical Features Count Plots) ---")
plt.figure(figsize=(18, 12))
for i, col in enumerate(categorical_features):
    plt.subplot(4, 4, i + 1)
    sns.countplot(x=df[col].dropna(), palette='viridis')
    plt.title(f'Count of {col}', fontsize=12)
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

print("\n--- Univariate Analysis (Target Variable Distribution) ---")
plt.figure(figsize=(5, 4))
sns.countplot(x=df[target_col], palette='coolwarm')
plt.title(f'Distribution of {target_col} (0: Not CKD, 1: CKD)', fontsize=14)
plt.xlabel('Class')
plt.ylabel('Count')
plt.xticks([0, 1], ['Not CKD', 'CKD'])
plt.show()
plt.show()
print(f"Distribution of target variable ('{target_col}'):")
print(df[target_col].value_counts())
print("Percentage distribution:") # No f-string formatting here
percentage_dist = (df[target_col].value_counts(normalize=True) * 100).round(2).astype(str) + '%'
print(percentage_dist)


# --- 6. Bivariate and Multivariate Analysis ---
print("\n--- Bivariate/Multivariate Analysis ---")

# Correlation Matrix for Numerical Features
temp_df_corr = df[numerical_features].fillna(df[numerical_features].median())
plt.figure(figsize=(12, 10))
sns.heatmap(temp_df_corr.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix of Numerical Features', fontsize=16)
plt.show()

# Numerical Features vs. Target Variable (Box Plots)
print("\n--- Numerical Features vs. Target Variable (Box Plots) ---")
plt.figure(figsize=(20, 15))
for i, col in enumerate(numerical_features):
    plt.subplot(4, 4, i + 1)
    sns.boxplot(x=df[target_col], y=df[col], palette='pastel')
    plt.title(f'{col} by {target_col}', fontsize=12)
    plt.xlabel(target_col)
    plt.ylabel(col)
plt.tight_layout()
plt.show()

# Categorical Features vs. Target Variable (Stacked Bar Plots)
print("\n--- Categorical Features vs. Target Variable (Stacked Bar Plots) ---")
plt.figure(figsize=(20, 15))
for i, col in enumerate(categorical_features):
    plt.subplot(4, 4, i + 1)
    ct = pd.crosstab(df[col], df[target_col], normalize='index') * 100
    ct.plot(kind='bar', stacked=True, ax=plt.gca(), cmap='viridis')
    plt.title(f'{col} vs. {target_col} (Stacked Bar - %)', fontsize=12)
    plt.xlabel(col)
    plt.ylabel('Percentage')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Class', labels=['Not CKD', 'CKD'])
plt.tight_layout()
plt.show()


# --- 7. Outlier Detection ---
print("\n--- Outlier Detection using Box Plots (Numerical Features) ---")
plt.figure(figsize=(20, 15))
for i, col in enumerate(numerical_features):
    plt.subplot(4, 4, i + 1)
    sns.boxplot(y=df[col].dropna(), color='lightcoral')
    plt.title(f'Box Plot of {col}', fontsize=12)
    plt.ylabel(col)
plt.tight_layout()
plt.show()

# --- 8. Data Cleaning & Feature Engineering (Consistent with app.py expectation) ---
print("\n--- Starting Data Cleaning and Feature Engineering (Aligned for app.py) ---")

# Step 8.1: Handle Inconsistent String Values specifically for categorical features
categorical_cols_to_clean = [
    'red_blood_cells', 'pus_cell', 'pus_cell_clumps', 'bacteria',
    'hypertension', 'diabetes_mellitus', 'coronary_artery_disease',
    'appetite', 'pedal_edema', 'anemia'
]

for col in categorical_cols_to_clean:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip().str.lower().replace({
            '\tyes': 'yes', ' yes': 'yes', 'yes ': 'yes',
            '\tno': 'no', ' no': 'no', 'no ': 'no',
            '\tnotpresent': 'notpresent', '\tpresent': 'present',
            '?': np.nan # Ensure any specific '?' are converted to NaN
        })
print("String inconsistencies in categorical features cleaned.")

# Step 8.2: Impute Missing Values (Median for numerical, Mode for categorical)
for col in numerical_features:
    if df[col].isnull().any():
        median_val = df[col].median()
        df[col].fillna(median_val, inplace=True)
        print(f"Missing values in numerical '{col}' imputed with median: {median_val}")

for col in categorical_features:
    if df[col].isnull().any():
        mode_val = df[col].mode()[0]
        df[col].fillna(mode_val, inplace=True)
        print(f"Missing values in categorical '{col}' imputed with mode: {mode_val}")

print("\nMissing values after imputation:")
print(df.isnull().sum()[df.isnull().sum() > 0])


# Step 8.3: One-Hot Encode Categorical Features (crucial for app.py compatibility)
X = df.drop(columns=[target_col])
y = df[target_col]

categorical_features_for_encoding = X.select_dtypes(include='object').columns.tolist()
X = pd.get_dummies(X, columns=categorical_features_for_encoding, drop_first=True)

print("\nCategorical features one-hot encoded using `pd.get_dummies` (drop_first=True).")
print(f"Shape of X after encoding: {X.shape}")
print("Sample of encoded features (X.head()):")
print(X.head())


# --- 10. Split the Data into Training and Testing Sets ---
print("\n--- Splitting data into Training and Testing Sets ---")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")


# --- 11. Train the Machine Learning Model (XGBoost Classifier) ---
print("\n--- Training Machine Learning Model (XGBoost Classifier) ---")
model = XGBClassifier(objective='binary:logistic', eval_metric='logloss', use_label_encoder=False, random_state=42)
model.fit(X_train, y_train)
print("XGBoost Classifier model trained successfully.")

# Evaluate model performance on the test set
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy on Test Set (XGBoost): {accuracy:.4f}")


# --- 12. Model Evaluation and Visualization ---
print("\n--- Model Evaluation and Visualizations ---")

# Feature Importance Plot
print("\n--- XGBoost Feature Importance ---")
plt.figure(figsize=(10, 6))
xgb_importance = model.feature_importances_
sorted_idx = np.argsort(xgb_importance)[::-1]
sorted_features = [X.columns[i] for i in sorted_idx]
plt.barh(sorted_features, xgb_importance[sorted_idx])
plt.xlabel('Feature Importance Score')
plt.title('XGBoost Feature Importance for CKD Prediction')
plt.gca().invert_yaxis()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# Confusion Matrix Heatmap
print("\n--- Confusion Matrix Heatmap ---")
cm = confusion_matrix(y_test, y_pred)
labels = ['Not CKD', 'CKD']

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title('Confusion Matrix Heatmap')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()

# Classification Report
print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred, target_names=labels))

# Final Accuracy Score
acc = accuracy_score(y_test, y_pred)
print(f"\n✅ Final Accuracy Score: {acc * 100:.2f}%")


# --- 13. Save the Trained Model and Feature Columns ---
model_filename = 'CKD.pkl'
with open(model_filename, 'wb') as file:
    pickle.dump(model, file)
print(f"\nModel saved as '{model_filename}' in the project root directory.")

feature_columns_filename = 'model_features.pkl'
with open(feature_columns_filename, 'wb') as file:
    pickle.dump(X.columns.tolist(), file)
print(f"Feature columns saved as '{feature_columns_filename}' in the project root directory.")

print("\n--- ML Pipeline Completed: Data processed, Model Trained (XGBoost), and Saved ---")