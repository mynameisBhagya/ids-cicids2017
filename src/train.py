from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    cohen_kappa_score,
    f1_score,
    matthews_corrcoef,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


from data import clean_dataset, find_label_column, load_dataset


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load and clean data
    data = load_dataset()
    data = clean_dataset(data)

    label_column = find_label_column(data)

    if label_column is None:
        raise ValueError("Target label column was not found.")

    # 2. Clean label text
    data[label_column] = (
        data[label_column]
        .astype(str)
        .str.strip()
    )

    print("\nOriginal labels:")
    print(data[label_column].value_counts())

    # 3. Convert the multiclass labels into two classes
    data["target"] = data[label_column].apply(
        lambda label: (
            "Benign"
            if label.strip().upper() == "BENIGN"
            else "Attack"
        )
    )

    # 4. Remove target and original label from the features
    X = data.drop(columns=[label_column, "target"])
    y = data["target"]

    # Use numerical features for the first baseline model
    X = X.select_dtypes(include=["number"]).copy()

    # Replace infinite values
    X = X.replace([float("inf"), float("-inf")], float("nan"))

    # Remove completely empty columns
    X = X.dropna(axis=1, how="all")

    if X.empty:
        raise ValueError("No usable numerical features were found.")

    # Remove obvious identifiers and fields that can create leakage
    suspicious_columns = [
        column for column in X.columns
        if any(
            term in column.lower()
            for term in [
                "flow id",
                "source ip",
                "destination ip",
                "timestamp",
            ]
        )
    ]

    if suspicious_columns:
        print("\nRemoving identifier/time columns:")
        print(suspicious_columns)
        X = X.drop(columns=suspicious_columns)

    if X.empty:
        raise ValueError("No features remain after column filtering.")

    # 5. Limit the initial experiment to a manageable sample
    # Later, evaluate the full dataset and rare attack classes.
    if len(X) > 100_000:
        X = X.sample(n=100_000, random_state=42)
        y = y.loc[X.index]

    print("\nFeatures used:", X.shape[1])
    print("Rows used:", len(X))
    print("\nBinary class distribution:")
    print(y.value_counts())

    # Check that both classes are present
    if y.nunique() != 2:
        raise ValueError(
            "Expected both Benign and Attack classes. "
            "Check the dataset labels."
        )

    # 6. Split into training and testing data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    # 7. Build a preprocessing and model pipeline
    preprocessing = ColumnTransformer(
        transformers=[
            (
                "numeric",
                SimpleImputer(strategy="median"),
                X.columns.tolist(),
            )
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=100,
                    random_state=42,
                    n_jobs=-1,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    # 8. Train the model
    print("\nTraining Random Forest...")
    model.fit(X_train, y_train)

    # 9. Predict on data not used for training
    predictions = model.predict(X_test)

    # 10. Evaluate performance
    print("\n===== BINARY CLASSIFICATION RESULTS =====")
    print("Accuracy:", accuracy_score(y_test, predictions))
    print("Macro F1:", f1_score(
        y_test, predictions, average="macro"
    ))
    print("MCC:", matthews_corrcoef(y_test, predictions))
    print("Cohen's kappa:", cohen_kappa_score(y_test, predictions))

    print("\nClassification report:")
    report = classification_report(
        y_test,
        predictions,
        output_dict=True,
        zero_division=0,
    )
    print(classification_report(
        y_test,
        predictions,
        zero_division=0,
    ))

    print("\nConfusion matrix:")
    print(confusion_matrix(
        y_test,
        predictions,
        labels=["Benign", "Attack"],
    ))

    # 11. Save the report
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(RESULTS_DIR / "binary_classification_report.csv")

    summary = pd.DataFrame(
        [{
            "model": "Random Forest",
            "accuracy": accuracy_score(y_test, predictions),
            "macro_f1": f1_score(
                y_test, predictions, average="macro"
            ),
            "mcc": matthews_corrcoef(y_test, predictions),
            "cohen_kappa": cohen_kappa_score(y_test, predictions),
            "rows": len(X),
            "features": X.shape[1],
            "random_seed": 42,
        }]
    )

    summary.to_csv(
        RESULTS_DIR / "binary_model_summary.csv",
        index=False,
    )

    print("\nResults saved in the results/ folder.")


if __name__ == "__main__":
    main()
