# Customer Churn Prediction Project

This repository demonstrates a complete machine learning project for predicting customer churn using scikit‑learn. The project follows best practices for reproducibility and includes an end-to-end workflow from data loading to model evaluation.

## Purpose

This project predicts whether customers will churn based on their usage patterns, demographics, and account information. The layout follows recommended directory organization for machine learning projects, making it easy to understand, maintain, and extend. All code is well documented and grouped by task.

## Project layout

```
.
├── main.py                 # Entry point that runs the entire pipeline
├── requirements.txt        # Python dependencies
├── data/
│   ├── processed/          # Created after running the pipeline
│   └── raw/
│       ├── train.csv       # Training dataset for churn prediction
│       └── test.csv        # Test dataset
├── notebooks/
│   ├── churn_analysis.ipynb
└── src/
    ├── data/
    │   ├── load_data.py
    │   ├── preprocess.py
    │   └── split_data.py
    ├── features/
    │   └── build_features.py
    ├── models/
    │   ├── train_model.py
    │   ├── dumb_model.py
    │   └── knn_model.py
    ├── utils/
    │   └── helper_functions.py
    └── visualization/
        ├── eda.py
        └── performance.py
```

`main.py` imports the modules inside `src/` and executes them to reproduce the analysis and results. Jupyter notebooks are provided only for prototyping and exploration—they are **not** meant to be the main entry point of the project.

## Dataset

The churn dataset contains customer information including:
- Demographics: Age, Gender
- Account details: Tenure, Subscription Type, Contract Length
- Usage patterns: Usage Frequency, Support Calls, Payment Delay
- Financial: Total Spend
- Target variable: Churn (0 = Not Churned, 1 = Churned)

## Running the project

Install the dependencies and run the pipeline. You should use the versions of the dependencies as specified by the requirements file:

```bash
conda create -n credit_fraud --file requirements.txt
conda activate credit_fraud
python main.py
```

This will:
1. Load the churn dataset from `data/raw/train.csv`
2. Perform exploratory data analysis (EDA) with visualizations
3. Clean and preprocess the data (handle missing values, encode categorical variables)
4. Split the data into training, validation, and test sets
5. Train a K-Nearest Neighbors (KNN) classifier and a baseline model
6. Evaluate both models on the validation set
7. Test the best-performing model on the test set
8. Display performance metrics including ROC curves and confusion matrices

The cleaned data will be written to `data/processed/` and all plots will be displayed interactively.
