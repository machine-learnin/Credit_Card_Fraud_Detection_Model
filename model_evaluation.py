# model_evaluation.py
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, confusion_matrix
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Create directory for reports if it doesn't exist
os.makedirs('reports/figures', exist_ok=True)

# Load test data
X_test = pd.read_csv('data/processed/X_test.csv')
y_test = pd.read_csv('data/processed/y_test.csv').values.ravel()

# Load models
models = {
    'Random Forest': joblib.load('models/random_forest_model.pkl'),
    'XGBoost': joblib.load('models/xgboost_model.pkl')
}

# Evaluate models
for name, model in models.items():
    print(f"\nEvaluating {name}...")
    y_pred = model.predict(X_test)

    # Metrics
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Classification Report:\n", classification_report(y_test, y_pred))
    print(f"ROC-AUC: {roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]):.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'reports/figures/confusion_matrix_{name.lower().replace(" ", "_")}.png')
    plt.show()

print("Model evaluation completed.")