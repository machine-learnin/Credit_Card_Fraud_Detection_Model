# data_preprocessing.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import joblib
import os

# Create directories for outputs if they don't exist
os.makedirs('data/processed', exist_ok=True)
os.makedirs('models', exist_ok=True)

# Load dataset
data = pd.read_csv('data/creditcard.csv')

# Check for missing values
print("Missing values in dataset:\n", data.isnull().sum())
if data.isnull().sum().sum() > 0:
    data = data.dropna()
    print("Dropped rows with missing values.")

# Separate features and target
X = data.drop('Class', axis=1)
y = data['Class']

# Scale Time and Amount
scaler = StandardScaler()
X[['Time', 'Amount']] = scaler.fit_transform(X[['Time', 'Amount']])

# Save scaler for deployment
joblib.dump(scaler, 'models/scaler.pkl')

# Apply SMOTE with a lower sampling strategy to avoid overfitting
smote = SMOTE(sampling_strategy=0.5, random_state=42)  # 1:2 ratio of fraud to non-fraud
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Save processed data
X_train.to_csv('data/processed/X_train.csv', index=False)
X_test.to_csv('data/processed/X_test.csv', index=False)
y_train.to_csv('data/processed/y_train.csv', index=False)
y_test.to_csv('data/processed/y_test.csv', index=False)

print("Data preprocessing completed. Training and test sets saved.")