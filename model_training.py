# model_training.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
import joblib

# Load processed data
X_train = pd.read_csv('data/processed/X_train.csv')
y_train = pd.read_csv('data/processed/y_train.csv').values.ravel()

# Define models and parameter grids
models = {
    'Random Forest': {
        'model': RandomForestClassifier(random_state=42),
        'params': {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5],
            'class_weight': [{0: 1, 1: 2}]  # Increase weight for fraud class
        }
    },
    'XGBoost': {
        'model': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
        'params': {
            'n_estimators': [100, 200],
            'max_depth': [3, 6],
            'learning_rate': [0.01, 0.1],
            'scale_pos_weight': [2]  # Increase weight for fraud class
        }
    }
}

# Train and tune models
for name, config in models.items():
    print(f"Training {name} with GridSearchCV...")
    grid_search = GridSearchCV(
        estimator=config['model'],
        param_grid=config['params'],
        cv=5,  # 5-fold cross-validation
        scoring='precision',  # Optimize for precision
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)

    # Save best model
    best_model = grid_search.best_estimator_
    joblib.dump(best_model, f'models/{name.lower().replace(" ", "_")}_model.pkl')
    print(f"Best parameters for {name}: {grid_search.best_params_}")
    print(f"Best precision score: {grid_search.best_score_:.4f}")

print("Model training completed.")