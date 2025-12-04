import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import os
from datetime import datetime

def load_preprocessed_data():
    """Load preprocessed data from pickle files."""
    X_train = joblib.load('X_train.pkl')
    X_val = joblib.load('X_val.pkl')
    X_test = joblib.load('X_test.pkl')
    y_train = joblib.load('y_train.pkl')
    y_val = joblib.load('y_val.pkl')
    y_test = joblib.load('y_test.pkl')
    vectorizer = joblib.load('vectorizer.pkl')

    return X_train, X_val, X_test, y_train, y_val, y_test, vectorizer

def train_logistic_regression(X_train, y_train, X_val, y_val):
    """Train Logistic Regression model."""
    print("Training Logistic Regression...")
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)

    # Validate
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    print(".3f")

    return model

def train_svm(X_train, y_train, X_val, y_val):
    """Train SVM model."""
    print("Training SVM...")
    model = SVC(kernel='linear', probability=True, random_state=42)
    model.fit(X_train, y_train)

    # Validate
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    print(".3f")

    return model

def train_random_forest(X_train, y_train, X_val, y_val):
    """Train Random Forest model."""
    print("Training Random Forest...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Validate
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    print(".3f")

    return model

def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate model on test set."""
    print(f"\nEvaluating {model_name} on test set:")
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(".3f")
    print(".3f")
    print(".3f")
    print(".3f")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['ham', 'spam']))

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def save_model(model, model_name, metrics):
    """Save trained model with metadata."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"{model_name}_{timestamp}.pkl"
    metadata_filename = f"{model_name}_{timestamp}_metadata.pkl"

    # Save model
    joblib.dump(model, model_filename)

    # Save metadata
    metadata = {
        'model_name': model_name,
        'timestamp': timestamp,
        'metrics': metrics,
        'model_file': model_filename
    }
    joblib.dump(metadata, metadata_filename)

    print(f"Model saved as: {model_filename}")
    print(f"Metadata saved as: {metadata_filename}")

    return model_filename, metadata_filename

def select_best_model(models_and_metrics):
    """Select the best performing model based on F1 score."""
    best_model = max(models_and_metrics, key=lambda x: x[1]['f1'])
    return best_model

def main():
    # Load data
    print("Loading preprocessed data...")
    X_train, X_val, X_test, y_train, y_val, y_test, vectorizer = load_preprocessed_data()

    # Train different models
    models = []

    # Logistic Regression
    lr_model = train_logistic_regression(X_train, y_train, X_val, y_val)
    lr_metrics = evaluate_model(lr_model, X_test, y_test, "Logistic Regression")
    models.append((lr_model, lr_metrics, "LogisticRegression"))

    # SVM
    svm_model = train_svm(X_train, y_train, X_val, y_val)
    svm_metrics = evaluate_model(svm_model, X_test, y_test, "SVM")
    models.append((svm_model, svm_metrics, "SVM"))

    # Random Forest
    rf_model = train_random_forest(X_train, y_train, X_val, y_val)
    rf_metrics = evaluate_model(rf_model, X_test, y_test, "Random Forest")
    models.append((rf_model, rf_metrics, "RandomForest"))

    # Select best model
    best_model, best_metrics, best_name = select_best_model(models)
    print(f"\nBest model: {best_name} with F1 score: {best_metrics['f1']:.3f}")

    # Save best model
    model_file, metadata_file = save_model(best_model, best_name, best_metrics)

    # Save vectorizer for inference
    joblib.dump(vectorizer, 'vectorizer.pkl')

    # Create current model symlink/reference
    with open('current_model.txt', 'w') as f:
        f.write(model_file)

    print(f"\nTraining complete! Best model saved as {model_file}")

if __name__ == "__main__":
    main()