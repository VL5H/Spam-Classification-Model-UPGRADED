import joblib
import numpy as np
from preprocess import clean_text
import os
from datetime import datetime

class SMSClassifier:
    def __init__(self, model_path=None, vectorizer_path='vectorizer.pkl'):
        """Initialize the SMS classifier with model and vectorizer."""
        self.model = None
        self.vectorizer = None
        self.model_path = model_path
        self.vectorizer_path = vectorizer_path
        self.load_model()

    def load_model(self):
        """Load the current model and vectorizer."""
        try:
            # Load current model reference
            if os.path.exists('current_model.txt'):
                with open('current_model.txt', 'r') as f:
                    model_file = f.read().strip()
                self.model_path = model_file

            if self.model_path and os.path.exists(self.model_path):
                self.model = joblib.load(self.model_path)
                print(f"Model loaded from {self.model_path}")
            else:
                print("No trained model found. Please train a model first.")
                return False

            if os.path.exists(self.vectorizer_path):
                self.vectorizer = joblib.load(self.vectorizer_path)
                print("Vectorizer loaded successfully")
            else:
                print("Vectorizer not found.")
                return False

            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    def preprocess_message(self, message):
        """Preprocess a single message for classification."""
        cleaned = clean_text(message)
        vectorized = self.vectorizer.transform([cleaned])
        return vectorized

    def classify_message(self, message):
        """Classify a message and return prediction with confidence."""
        if self.model is None or self.vectorizer is None:
            return {"error": "Model not loaded"}

        try:
            # Preprocess message
            X = self.preprocess_message(message)

            # Get prediction and probabilities
            prediction = self.model.predict(X)[0]
            probabilities = self.model.predict_proba(X)[0]

            # Calculate confidence (max probability)
            confidence = float(max(probabilities))

            # Convert prediction to label
            label = "spam" if prediction == 1 else "ham"

            result = {
                "message": message,
                "classification": label,
                "confidence": confidence,
                "probabilities": {
                    "ham": float(probabilities[0]),
                    "spam": float(probabilities[1])
                },
                "needs_feedback": confidence < 0.8  # Threshold for feedback
            }

            return result

        except Exception as e:
            return {"error": f"Classification failed: {str(e)}"}

    def get_model_info(self):
        """Get information about the current model."""
        if self.model is None:
            return {"error": "No model loaded"}

        try:
            # Try to load metadata
            metadata_file = self.model_path.replace('.pkl', '_metadata.pkl')
            if os.path.exists(metadata_file):
                metadata = joblib.load(metadata_file)
                return {
                    "model_file": self.model_path,
                    "model_name": metadata.get('model_name', 'Unknown'),
                    "timestamp": metadata.get('timestamp', 'Unknown'),
                    "metrics": metadata.get('metrics', {}),
                    "loaded": True
                }
            else:
                return {
                    "model_file": self.model_path,
                    "model_name": "Unknown",
                    "timestamp": "Unknown",
                    "loaded": True
                }
        except Exception as e:
            return {"error": f"Could not get model info: {str(e)}"}

def test_inference():
    """Test the inference engine with sample messages."""
    classifier = SMSClassifier()

    if classifier.model is None:
        print("No model available for testing.")
        return

    # Test messages
    test_messages = [
        "Hey, how are you doing today?",
        "WINNER! You have won a free iPhone! Click here to claim your prize!",
        "Meeting scheduled for tomorrow at 3 PM",
        "URGENT: Your account has been suspended. Call now to reactivate!",
        "Thanks for your help with the project",
        "FREE entry into our £250 weekly competition just text YES to 80082"
    ]

    print("Testing SMS Classification:")
    print("=" * 50)

    for msg in test_messages:
        result = classifier.classify_message(msg)
        if "error" not in result:
            print(f"Message: {msg[:50]}...")
            print(f"Classification: {result['classification'].upper()}")
            print(".3f")
            print(f"Needs feedback: {result['needs_feedback']}")
            print("-" * 30)
        else:
            print(f"Error: {result['error']}")

if __name__ == "__main__":
    test_inference()