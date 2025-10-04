import argparse
import joblib
import os
import pandas as pd
from scipy.sparse import hstack

from data_processing import clean_text, extract_numerical_features, NUMERICAL_COLS


def predict(text, vectorizer, scaler, model):
    """Predict if a text is spam or not, and return the probability."""
    cleaned_text = clean_text(text)
    numerical_features = extract_numerical_features(cleaned_text)
    
    numerical_df = pd.DataFrame([numerical_features], columns=NUMERICAL_COLS)

    text_vector = vectorizer.transform([cleaned_text])
    numerical_vector = scaler.transform(numerical_df.values)

    combined_features = hstack([text_vector, numerical_vector])

    prediction = model.predict(combined_features)[0]
    label = "spam" if prediction == 1 else "ham"
    
    probability = 0.0
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(combined_features)[0]
        probability = probs[prediction]

    return label, probability

def main():
    """
    Load models and predict text from a file.
    """

    artifacts_dir = "artifacts"
    models_dir = "models"
    prefix = "spam"

    parser = argparse.ArgumentParser(description="Predict if the content of a .txt file is spam or ham.")
    parser.add_argument("file_path", type=str, help="Path to the .txt file to be evaluated.")
    args = parser.parse_args()

    # Load Artifacts
    vectorizer_path = os.path.join(artifacts_dir, f"{prefix}_vectorizer.joblib")
    scaler_path = os.path.join(artifacts_dir, f"{prefix}_scaler.joblib")
    try:
        vectorizer = joblib.load(vectorizer_path)
        scaler = joblib.load(scaler_path)
    except FileNotFoundError as e:
        print(f"Error: Artifact not found. {e}")
        print("Please ensure you have run the training script (`python main.py train`) to generate the artifacts.")
        return

    # Load Models
    model_files = {
        "Logistic Regression": os.path.join(models_dir, "logreg.joblib"),
        "Naive Bayes": os.path.join(models_dir, "nb.joblib"),
        "Random Forest": os.path.join(models_dir, "rf.joblib")
    }

    models = {}
    all_models_found = True
    for name, path in model_files.items():
        try:
            models[name] = joblib.load(path)
        except FileNotFoundError:
            print(f"Error: Model '{name}' not found at '{path}'.")
            all_models_found = False
    
    if not all_models_found:
        print("Please ensure you have run the training script (`python main.py train`) to generate the models.")
        return

    # Read Input Text
    try:
        with open(args.file_path, 'r', encoding='utf-8') as f:
            text_to_predict = f.read()
    except FileNotFoundError:
        print(f"Error: Input file not found at '{args.file_path}'.")
        return
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return

    # Make and Display Predictions
    print(f"Analyzing text from: '{args.file_path}'\n")
    print("--- Text Content ---")
    print(text_to_predict)
    print("--------------------\n")
    print("--- Predictions ---")
    
    for name, model in models.items():
        label, prob = predict(text_to_predict, vectorizer, scaler, model)
        print(f"- {name}: {label.upper()} (Confidence: {prob:.2%})")
    print("-------------------\n")


if __name__ == "__main__":
    main()
