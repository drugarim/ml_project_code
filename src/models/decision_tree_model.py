import pandas as pd
from sklearn.tree import DecisionTreeClassifier


def train_decision_tree_model(X_train: pd.DataFrame, y_train: pd.Series) -> DecisionTreeClassifier:
    """Train and return a Decision Tree classifier.
    
    Args:
        X_train: Training features
        y_train: Training target variable
        
    Returns:
        Trained DecisionTreeClassifier model
    """
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model
