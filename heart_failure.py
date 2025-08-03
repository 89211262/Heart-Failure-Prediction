import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve

# Load data
df = pd.read_csv(r'C:\Users\jorda\OneDrive\Desktop\heart failure prediction\Heart-Failure-Prediction\data\heart.csv', sep=';')
print("First 5 rows:\n", df.head())

# Data info
print("\nData Info:")
print(df.info())
print("\nSummary Stats:")
print(df.describe())
print("\nMissing Values:\n", df.isnull().sum())

# Drop duplicates
df = df.drop_duplicates()

# Display basic info before cleaning
print("Original shape:", df.shape)

# Function to remove outliers using IQR
def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# Remove outliers from all numerical columns
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns

for col in numerical_cols:
    original_len = len(df)
    df = remove_outliers_iqr(df, col)
    print(f"Removed outliers from {col}, new shape: {df.shape} (removed {original_len - len(df)} rows)")

# Save cleaned dataset to the specified directory
output_path = r"C:\Users\jorda\OneDrive\Desktop\heart failure prediction\Heart-Failure-Prediction\data\heart_cleaned.csv"
df.to_csv(output_path, index=False)
print(f"Cleaned data saved to: {output_path}")

# Check categorical unique values
for col in ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']:
    print(f"Unique values in {col}: {df[col].unique()}")

# Label encode binary categorical features
label_enc_cols = ['Sex', 'ExerciseAngina']
le = LabelEncoder()
for col in label_enc_cols:
    df[col] = le.fit_transform(df[col])

# One-hot encode multi-category features
df_encoded = pd.get_dummies(df, columns=['ChestPainType', 'RestingECG', 'ST_Slope'], drop_first=False)

# === REMOVE OUTLIERS ===
def remove_outliers_iqr(dataframe, cols):
    for col in cols:
        Q1 = dataframe[col].quantile(0.25)
        Q3 = dataframe[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        dataframe = dataframe[(dataframe[col] >= lower_bound) & (dataframe[col] <= upper_bound)]
    return dataframe

# Apply to all numeric columns
numeric_cols = df_encoded.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols.remove('HeartDisease')  # Do not remove target labels
df_encoded = remove_outliers_iqr(df_encoded, numeric_cols)

# Separate features and target
X = df_encoded.drop('HeartDisease', axis=1)
y = df_encoded['HeartDisease']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

### LOGISTIC REGRESSION
logreg = LogisticRegression(max_iter=2000, random_state=42)
logreg.fit(X_train_scaled, y_train)
y_pred_logreg = logreg.predict(X_test_scaled)

print("\n=== Logistic Regression ===")
print("Accuracy:", accuracy_score(y_test, y_pred_logreg))
print(classification_report(y_test, y_pred_logreg))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_logreg))
print("ROC AUC Score:", roc_auc_score(y_test, logreg.predict_proba(X_test_scaled)[:, 1]))

### RANDOM FOREST
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train)
y_pred_rf = rf.predict(X_test_scaled)

print("\n=== Random Forest ===")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))
print("ROC AUC Score:", roc_auc_score(y_test, rf.predict_proba(X_test_scaled)[:, 1]))

# Feature Importance Plot
feature_importances = pd.Series(rf.feature_importances_, index=X.columns)
plt.figure(figsize=(10, 6))
feature_importances.nlargest(10).sort_values().plot(kind='barh', color='teal')
plt.title('Top 10 Feature Importances (Random Forest)')
plt.tight_layout()
plt.show()

### ARTIFICIAL NEURAL NETWORK
ann_model = MLPClassifier(
    hidden_layer_sizes=(32, 16),
    max_iter=500,
    learning_rate_init=0.001,
    early_stopping=True,
    random_state=42
)
ann_model.fit(X_train_scaled, y_train)
y_pred_ann = ann_model.predict(X_test_scaled)

print("\n=== ANN (MLPClassifier) ===")
print("Accuracy:", accuracy_score(y_test, y_pred_ann))
print(classification_report(y_test, y_pred_ann))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_ann))
print("ROC AUC Score:", roc_auc_score(y_test, ann_model.predict_proba(X_test_scaled)[:, 1]))

# ROC Curve comparison
plt.figure(figsize=(8, 6))
models = {
    "Logistic Regression": logreg,
    "Random Forest": rf,
    "ANN": ann_model
}

for name, model in models.items():
    probs = model.predict_proba(X_test_scaled)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, probs)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc_score(y_test, probs):.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.tight_layout()
plt.show()

# Save final cleaned version with no outliers
df_encoded.to_csv('heart_failure_processed.csv', index=False)
