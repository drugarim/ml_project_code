import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def train_random_forest_model(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
    """Train and return a Random Forest classifier.
    
    Args:
        X_train: Training features
        y_train: Training target variable
        
    Returns:
        Trained RandomForestClassifier model
    """
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model
