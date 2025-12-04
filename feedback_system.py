import os
import json
from datetime import datetime, timedelta
from inference_engine import SMSClassifier
import joblib

class FeedbackSystem:
    def __init__(self, feedback_log='user_feedback.log'):
        self.feedback_log = feedback_log
        self.classifier = SMSClassifier()

    def log_feedback(self, message, predicted_label, correct_label, confidence):
        """Log user feedback to the feedback file."""
        timestamp = datetime.now().isoformat()

        feedback_entry = {
            "timestamp": timestamp,
            "message": message,
            "predicted": predicted_label,
            "correct": correct_label,
            "confidence": confidence,
            "was_correct": predicted_label == correct_label
        }

        # Append to log file
        with open(self.feedback_log, 'a', encoding='utf-8') as f:
            f.write(json.dumps(feedback_entry) + '\n')

        print(f"Feedback logged: {predicted_label} -> {correct_label}")

    def collect_feedback(self, message, predicted_label, confidence):
        """Collect feedback from user for low confidence predictions."""
        print(f"\nModel classified: '{message[:50]}...' as {predicted_label.upper()}")
        print(".3f")

        while True:
            response = input("Is this correct? (y/n/skip): ").lower().strip()
            if response in ['y', 'yes']:
                self.log_feedback(message, predicted_label, predicted_label, confidence)
                return True
            elif response in ['n', 'no']:
                correct_label = "spam" if predicted_label == "ham" else "ham"
                self.log_feedback(message, predicted_label, correct_label, confidence)
                return False
            elif response in ['s', 'skip']:
                print("Feedback skipped.")
                return None
            else:
                print("Please enter 'y' (yes), 'n' (no), or 's' (skip).")

    def get_feedback_prompt(self, message, predicted_label, confidence):
        """Return feedback prompt data for GUI instead of collecting input."""
        return {
            "message": message,
            "predicted": predicted_label,
            "confidence": confidence,
            "prompt_text": f"Model classified: '{message[:50]}...' as {predicted_label.upper()}\nConfidence: {confidence:.3f}\n\nIs this correct?",
            "options": ["Yes", "No", "Skip"]
        }

    def submit_feedback(self, message, predicted_label, confidence, user_response):
        """Submit feedback based on user response from GUI."""
        if user_response == "Yes":
            self.log_feedback(message, predicted_label, predicted_label, confidence)
            return True
        elif user_response == "No":
            correct_label = "spam" if predicted_label == "ham" else "ham"
            self.log_feedback(message, predicted_label, correct_label, confidence)
            return False
        elif user_response == "Skip":
            return None
        else:
            raise ValueError("Invalid response. Must be 'Yes', 'No', or 'Skip'")

    def load_feedback_data(self):
        """Load feedback data from log file."""
        feedback_data = []
        if os.path.exists(self.feedback_log):
            with open(self.feedback_log, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        feedback_data.append(entry)
                    except json.JSONDecodeError:
                        continue
        return feedback_data

    def update_training_dataset(self, original_dataset='SMSSpamCollection.txt'):
        """Update the original dataset with feedback data."""
        # Load original dataset
        original_data = []
        with open(original_dataset, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    label, message = parts
                    original_data.append((label, message))

        # Load feedback data
        feedback_data = self.load_feedback_data()

        # Convert feedback to same format
        for entry in feedback_data:
            label = 'spam' if entry['correct'] == 'spam' else 'ham'
            message = entry['message']
            original_data.append((label, message))

        # Save updated dataset
        updated_file = f"{original_dataset.rsplit('.', 1)[0]}_updated.txt"
        with open(updated_file, 'w', encoding='utf-8') as f:
            for label, message in original_data:
                f.write(f"{label}\t{message}\n")

        print(f"Updated dataset saved as: {updated_file}")
        print(f"Original samples: {len(original_data) - len(feedback_data)}")
        print(f"Feedback samples added: {len(feedback_data)}")
        print(f"Total samples: {len(original_data)}")

        return updated_file

    def retrain_model(self, updated_dataset=None):
        """Retrain the model with updated dataset."""
        if updated_dataset is None:
            updated_dataset = self.update_training_dataset()

        # Import and run preprocessing
        from preprocess import load_dataset, preprocess_data, balance_dataset, vectorize_text, split_data

        print("Retraining model with updated dataset...")

        # Load and preprocess updated data
        df = load_dataset(updated_dataset)
        df = preprocess_data(df)
        df_balanced = balance_dataset(df)
        X, y, vectorizer = vectorize_text(df_balanced)
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

        # Train SVM (best performing model)
        from sklearn.svm import SVC
        model = SVC(kernel='linear', probability=True, random_state=42)
        model.fit(X_train, y_train)

        # Evaluate
        from sklearn.metrics import accuracy_score, f1_score
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        print(".3f")
        print(".3f")

        # Save retrained model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"SVM_retrained_{timestamp}.pkl"
        joblib.dump(model, model_filename)

        # Update vectorizer
        joblib.dump(vectorizer, 'vectorizer.pkl')

        # Update current model reference
        with open('current_model.txt', 'w') as f:
            f.write(model_filename)

        print(f"Retrained model saved as: {model_filename}")

        return model_filename

    def cleanup_old_feedback(self, days=30):
        """Remove feedback entries older than specified days."""
        if not os.path.exists(self.feedback_log):
            return

        cutoff_date = datetime.now() - timedelta(days=days)
        feedback_data = self.load_feedback_data()

        # Filter out old entries
        recent_feedback = []
        removed_count = 0

        for entry in feedback_data:
            entry_date = datetime.fromisoformat(entry['timestamp'])
            if entry_date > cutoff_date:
                recent_feedback.append(entry)
            else:
                removed_count += 1

        # Rewrite log file with recent entries only
        with open(self.feedback_log, 'w', encoding='utf-8') as f:
            for entry in recent_feedback:
                f.write(json.dumps(entry) + '\n')

        if removed_count > 0:
            print(f"Cleaned up {removed_count} old feedback entries (>30 days)")

def test_feedback_system():
    """Test the feedback system."""
    feedback_sys = FeedbackSystem()

    # Test classification and feedback collection
    test_message = "Congratulations! You've won a free vacation!"

    result = feedback_sys.classifier.classify_message(test_message)
    if "error" not in result:
        print(f"Testing feedback for: {test_message}")
        print(f"Predicted: {result['classification']} (confidence: {result['confidence']:.3f})")

        # Simulate user feedback
        feedback_sys.log_feedback(test_message, result['classification'], "spam", result['confidence'])
        print("Feedback logged successfully")

    # Show feedback stats
    feedback_data = feedback_sys.load_feedback_data()
    print(f"\nTotal feedback entries: {len(feedback_data)}")

    if feedback_data:
        correct_predictions = sum(1 for entry in feedback_data if entry['was_correct'])
        accuracy = correct_predictions / len(feedback_data)
        print(".3f")

if __name__ == "__main__":
    test_feedback_system()