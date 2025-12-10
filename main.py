from src.data.load_data import load_dataset
from src.data.preprocess import clean_dataset
from src.visualization.eda import plot_eda
from src.models.train_model import split_data, plot_roc_curve
from src.models.knn_model import train_knn_model
from src.models.dumb_model import train_dumb_model
from src.models.decision_tree_model import train_decision_tree_model
from src.models.random_forest_model import train_random_forest_model
from src.visualization.performance import (
    plot_confusion_matrices,
    plot_performance_comparison,
)
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns


def main() -> None:
    print("---Loading data...")
    raw_df = load_dataset("data/raw/train.csv")
    
    # Print shape of the raw dataset
    print(f"Raw dataset shape: {raw_df.shape}")
    
    # Remove missing values for EDA
    raw_df_clean = raw_df.dropna()

    print("---Creating EDA visuals...")
    plot_eda(raw_df_clean)

    print("---Cleaning data...")
    clean_df = clean_dataset(raw_df)

    print(f"Cleaned dataset shape: {clean_df.shape}")

    print("---Splitting data...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(clean_df)

    print("---Training models...")
    knn_model = train_knn_model(X_train, y_train)
    dumb_model = train_dumb_model(X_train, y_train)
    dt_model = train_decision_tree_model(X_train, y_train)
    rf_model = train_random_forest_model(X_train, y_train)

    print("---Evaluating on validation set...")
    y_val_pred_knn = knn_model.predict(X_val)
    y_val_pred_dumb = dumb_model.predict(X_val)
    y_val_pred_dt = dt_model.predict(X_val)
    y_val_pred_rf = rf_model.predict(X_val)

    val_prob_knn = knn_model.predict_proba(X_val)[:, 1]
    val_prob_dumb = dumb_model.predict_proba(X_val)[:, 1]
    val_prob_dt = dt_model.predict_proba(X_val)[:, 1]
    val_prob_rf = rf_model.predict_proba(X_val)[:, 1]

    # Print validation metrics for all models
    print("\n--- Validation Set Metrics ---")
    models = {
        "K-NN": (y_val_pred_knn, val_prob_knn),
        "Never Churn": (y_val_pred_dumb, val_prob_dumb),
        "Decision Tree": (y_val_pred_dt, val_prob_dt),
        "Random Forest": (y_val_pred_rf, val_prob_rf)
    }
    
    model_scores = {}
    for model_name, (y_pred, y_prob) in models.items():
        acc = accuracy_score(y_val, y_pred)
        prec = precision_score(y_val, y_pred, zero_division=0)
        rec = recall_score(y_val, y_pred, zero_division=0)
        f1 = f1_score(y_val, y_pred, zero_division=0)
        model_scores[model_name] = acc
        print(f"{model_name:20} - Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")

    plot_confusion_matrices(y_val, y_val_pred_dumb, y_val_pred_knn)
    plot_performance_comparison(y_val, y_val_pred_dumb, y_val_pred_knn)

    auc_dumb = plot_roc_curve(y_val, val_prob_dumb, "Never Churn")
    auc_knn = plot_roc_curve(y_val, val_prob_knn, "3-NN")
    auc_dt = plot_roc_curve(y_val, val_prob_dt, "Decision Tree")
    auc_rf = plot_roc_curve(y_val, val_prob_rf, "Random Forest")

    # Find best model based on validation AUC
    auc_scores = {
        "K-NN": auc_knn,
        "Never Churn": auc_dumb,
        "Decision Tree": auc_dt,
        "Random Forest": auc_rf
    }
    best_label = max(auc_scores, key=auc_scores.get)
    best_auc = auc_scores[best_label]
    
    model_map = {
        "K-NN": knn_model,
        "Never Churn": dumb_model,
        "Decision Tree": dt_model,
        "Random Forest": rf_model
    }
    best_model = model_map[best_label]

    print(f"\n---Best Model: {best_label} (AUC: {best_auc:.4f})---")
    print(f"---Testing best model ({best_label})...")
    y_test_pred = best_model.predict(X_test)
    test_prob = best_model.predict_proba(X_test)[:, 1]
    plot_roc_curve(y_test, test_prob, f"Test {best_label}")

    cm = confusion_matrix(y_test, y_test_pred)
    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Best Model Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

    print("Done.")


if __name__ == "__main__":
    main()
