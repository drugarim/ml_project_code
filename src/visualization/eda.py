import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_eda(df: pd.DataFrame) -> None:
    """Create exploratory plots including bivariate views colored by churn."""
    sns.countplot(x='Churn', data=df)
    plt.title('Churn Class Distribution')
    plt.show()

    sns.countplot(x='Subscription Type', data=df)
    plt.title('Subscription Type Distribution')
    plt.show()

    sns.countplot(x='Gender', data=df)
    plt.title('Gender Distribution')
    plt.show()

    sns.histplot(df['Tenure'], bins=30)
    plt.title('Tenure Distribution')
    plt.show()

    # Bivariate visualizations colored by churn
    sns.countplot(data=df, x='Subscription Type', hue='Churn', palette=['green', 'red'])
    plt.title('Subscription Type vs Churn')
    plt.xlabel('Subscription Type')
    plt.ylabel('Count')
    plt.legend(title='Churn', labels=['Not Churned', 'Churned'])
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=df,
        x='Tenure',
        y='Usage Frequency',
        hue='Churn',
        palette={0: 'green', 1: 'red'},
    )
    plt.title(
        'Churn vs Not Churned: Tenure vs Usage Frequency'
    )
    plt.xlabel('Tenure')
    plt.ylabel('Usage Frequency')
    plt.show()


if __name__ == "__main__":
    from src.data.load_data import load_dataset
    from src.data.preprocess import clean_dataset

    raw = load_dataset("data/raw/card_transdata.csv")
    clean = clean_dataset(raw)
    plot_eda(clean)
