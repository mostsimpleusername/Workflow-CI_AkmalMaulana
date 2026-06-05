import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV
)

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# Experiment
mlflow.set_experiment(
    "Telco_Customer_Churn_Tuning"
)

# Load Dataset
df = pd.read_csv(
    "customer_churn_processed.csv"
)

X = df.drop(
    "Churn",
    axis=1
)

y = df["Churn"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Hyperparameter Tuning
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [5, 10, None],
    "min_samples_split": [2, 5]
}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(
        random_state=42
    ),
    param_grid=param_grid,
    cv=3,
    scoring="accuracy",
    n_jobs=-1
)

with mlflow.start_run():

    grid_search.fit(
        X_train,
        y_train
    )

    best_model = grid_search.best_estimator_

    predictions = best_model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions
    )

    recall = recall_score(
        y_test,
        predictions
    )

    f1 = f1_score(
        y_test,
        predictions
    )

    # Manual Logging
    mlflow.log_params(
        grid_search.best_params_
    )

    mlflow.log_metric(
        "accuracy",
        accuracy
    )

    mlflow.log_metric(
        "precision",
        precision
    )

    mlflow.log_metric(
        "recall",
        recall
    )

    mlflow.log_metric(
        "f1_score",
        f1
    )

    mlflow.sklearn.log_model(
        best_model,
        artifact_path="model"
    )

    print(
        "Best Parameters:",
        grid_search.best_params_
    )

    print(
        f"Accuracy: {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall: {recall:.4f}"
    )

    print(
        f"F1 Score: {f1:.4f}"
    )