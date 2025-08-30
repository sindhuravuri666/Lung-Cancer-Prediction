import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from app.core.config import settings


def main():
    # ---------------- Load data ----------------
    data_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw", "survey_lung_cancer.csv")
    # Clean up column names (strip whitespace)
    

    data_path = os.path.abspath(data_path)
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}")
    df = pd.read_csv(data_path)
    df.columns = df.columns.str.strip()

    # Encode target (YES=1, NO=0)
    df["LUNG_CANCER"] = df["LUNG_CANCER"].map({"YES": 1, "NO": 0})

    X = df.drop("LUNG_CANCER", axis=1)
    y = df["LUNG_CANCER"]

    # ---------------- Preprocessing ----------------
    categorical_features = ["GENDER"]
    numeric_features = [col for col in X.columns if col not in categorical_features]

    categorical_transformer = OneHotEncoder(drop="first")  # encode M/F
    numeric_transformer = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", categorical_transformer, categorical_features),
            ("numeric", numeric_transformer, numeric_features),
        ]
    )

    # ---------------- Models ----------------
    models = {
        "log_reg": LogisticRegression(max_iter=1000),
        "random_forest": RandomForestClassifier(n_estimators=200, random_state=42),
    }

    best_model = None
    best_score = 0.0

    for name, model in models.items():
        pipe = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", model)])
        scores = cross_val_score(pipe, X, y, cv=5, scoring="accuracy")
        mean_score = scores.mean()
        print(f"{name} CV accuracy: {mean_score:.3f}")

        if mean_score > best_score:
            best_score = mean_score
            best_model = pipe

    # ---------------- Train/Test Split ----------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Final test accuracy: {acc:.3f}")

    # ---------------- Save Model ----------------
    os.makedirs(os.path.dirname(settings.MODEL_PATH), exist_ok=True)
    joblib.dump(best_model, settings.MODEL_PATH)
    print(f"Model saved to {settings.MODEL_PATH}")


if __name__ == "__main__":
    main()
