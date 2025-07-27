# eda.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create directory for reports if it doesn't exist
os.makedirs('reports/figures', exist_ok=True)

# Load dataset
data = pd.read_csv('data/creditcard.csv')

# Class distribution
print("Class distribution:\n", data['Class'].value_counts(normalize=True))
plt.figure(figsize=(6, 4))
sns.countplot(x='Class', data=data)
plt.title('Class Distribution (0: Non-Fraud, 1: Fraud)')
plt.savefig('reports/figures/class_distribution.png')
plt.show()

# Amount distribution by class
plt.figure(figsize=(8, 4))
sns.boxplot(x='Class', y='Amount', data=data)
plt.title('Transaction Amount by Class')
plt.savefig('reports/figures/amount_by_class.png')
plt.show()

print("EDA completed. Data visualizations saved to reports/figures/.")