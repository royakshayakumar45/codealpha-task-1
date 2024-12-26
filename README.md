# codealpha-task-1import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Step 2: Data Preparation
# Load the dataset (replace 'credit_data.csv' with your dataset file)
data = pd.read_csv('credit_data.csv')

# Example feature engineering
data['debt_to_income_ratio'] = data['debt'] / data['income']
data = data.dropna()  # Handle missing values

# Splitting features and target
X = data.drop(['creditworthiness'], axis=1)
y = data['creditworthiness']

# Encode categorical features
categorical_features = X.select_dtypes(include=['object']).columns
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
X_encoded = pd.DataFrame(encoder.fit_transform(X[categorical_features]))
X_encoded.columns = encoder.get_feature_names_out(categorical_features)

# Combine numerical and encoded categorical features
numerical_features = X.select_dtypes(exclude=['object'])
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(numerical_features), columns=numerical_features.columns)
X_prepared = pd.concat([X_scaled, X_encoded], axis=1)

# Split the dataset
X_train, X_temp, y_train, y_temp = train_test_split(X_prepared, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Step 3: Model Development
# Train a Random Forest Classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Step 4: Training & Tuning
# Evaluate on validation set
val_predictions = model.predict(X_val)
val_probs = model.predict_proba(X_val)[:, 1]

print("Validation Metrics:")
print("Accuracy:", accuracy_score(y_val, val_predictions))
print("Precision:", precision_score(y_val, val_predictions))
print("Recall:", recall_score(y_val, val_predictions))
print("F1 Score:", f1_score(y_val, val_predictions))
print("ROC AUC:", roc_auc_score(y_val, val_probs))

# Step 5: Model Evaluation
# Final evaluation on test data
test_predictions = model.predict(X_test)
test_probs = model.predict_proba(X_test)[:, 1]

print("\nTest Metrics:")
print("Accuracy:", accuracy_score(y_test, test_predictions))
print("Precision:", precision_score(y_test, test_predictions))
print("Recall:", recall_score(y_test, test_predictions))
print("F1 Score:", f1_score(y_test, test_predictions))
print("ROC AUC:", roc_auc_score(y_test, test_probs))

# Save the trained model (optional)
import joblib
joblib.dump(model, 'credit_scoring_model.pkl')

new repo
