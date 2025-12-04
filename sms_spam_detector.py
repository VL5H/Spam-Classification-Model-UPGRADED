import os
import sys
from datetime import datetime
from inference_engine import SMSClassifier
from feedback_system import FeedbackSystem
from logging_system import logger, echo_logger
import joblib
import time
import json

class SMSSpamDetector:
    def __init__(self):
        self.classifier = SMSClassifier()
        self.feedback_system = FeedbackSystem()
        self.current_model = None
        self.load_current_model()
        self.setup_echo_logging()

    def load_current_model(self):
        """Load the current model on startup."""
        try:
            if os.path.exists('current_model.txt'):
                with open('current_model.txt', 'r') as f:
                    model_file = f.read().strip()
                self.current_model = model_file
                logger.log_model_action("MODEL_LOADED", f"Loaded {model_file} on startup")
            else:
                print("No saved model found. Please train a model first.")
                logger.log_error("NO_MODEL_FOUND", "No current_model.txt file found on startup")
        except Exception as e:
            logger.log_error("MODEL_LOAD_ERROR", f"Failed to load model on startup: {str(e)}")

    def setup_echo_logging(self):
        """Set up echo logging and prompt user on startup."""
        print("\n" + "="*50)
        print("USER ECHO LOG SYSTEM")
        print("="*50)

        # Show current state
        if echo_logger.echo_enabled:
            print("Echo logging is currently ENABLED")
            print(f"Current session: {echo_logger.current_session}")
        else:
            print("Echo logging is currently DISABLED")

        # Prompt user
        while True:
            response = input("Would you like to enable User Echo Log? (y/n): ").strip().lower()
            if response in ['y', 'yes']:
                if echo_logger.enable_echo_log():
                    print("User Echo Log has been ENABLED.")
                    print(f"Session {echo_logger.current_session} started.")
                else:
                    print("User Echo Log is already enabled.")
                break
            elif response in ['n', 'no']:
                if echo_logger.echo_enabled:
                    echo_logger.disable_echo_log()
                    print("User Echo Log has been DISABLED.")
                else:
                    print("User Echo Log remains DISABLED.")
                break
            else:
                print("Please enter 'y' for yes or 'n' for no.")

        print("="*50)

    def display_menu(self):
        """Display the main menu."""
        print("\n" + "="*50)
        print("    SMS SPAM DETECTION SYSTEM")
        print("="*50)
        echo_status = "ENABLED" if echo_logger.echo_enabled else "DISABLED"
        print(f"Echo Log Status: {echo_status}")
        print("1. Classify - Classify a message as Spam or Normal")
        print("2. Feedback - Provide feedback on the latest classification")
        print("3. Manage Models - View, create, load, copy, or delete model versions")
        print("4. Update - Update training dataset with user feedback")
        print("5. Retrain - Retrain the current model on updated dataset")
        print("6. Stats - Display last 5 log entries from each log file")
        print("7. Benchmark - Run performance benchmarking on current model")
        print("8. Help - Show command definitions")
        print("9. Quit - Save current model and quit")
        print("="*50)

    def classify_message(self):
        """Classify a message and handle feedback if needed."""
        if not self.classifier.model:
            print("No model loaded. Please load or train a model first.")
            return

        message = input("Enter the message to classify: ").strip()
        echo_logger.log_interaction("USER_INPUT", message)

        if not message:
            print("Message cannot be empty.")
            return

        logger.log_model_action("CLASSIFICATION_REQUEST", f"Message length: {len(message)}")

        result = self.classifier.classify_message(message)

        if "error" in result:
            print(f"Classification error: {result['error']}")
            logger.log_error("CLASSIFICATION_ERROR", result['error'])
            return

        # Display result
        output_lines = [
            f"\nClassification: {result['classification'].upper()}",
            f"Confidence: {result['confidence']:.3f}"
        ]

        for line in output_lines:
            print(line)
            echo_logger.log_interaction("SYSTEM_OUTPUT", line)

        # Log classification
        logger.log_model_action("MESSAGE_CLASSIFIED",
                                f"Result: {result['classification']}, Confidence: {result['confidence']:.3f}")

        # Handle feedback if confidence is low
        if result['needs_feedback']:
            feedback_msg = "\nModel confidence is low. Your feedback will help improve the model."
            print(feedback_msg)
            echo_logger.log_interaction("SYSTEM_OUTPUT", feedback_msg)

            feedback_result = self.feedback_system.collect_feedback(
                message, result['classification'], result['confidence']
            )
            if feedback_result is not None:
                thank_msg = "Thank you for your feedback!"
                print(thank_msg)
                echo_logger.log_interaction("SYSTEM_OUTPUT", thank_msg)
        else:
            no_feedback_msg = "High confidence classification - no feedback needed."
            print(no_feedback_msg)
            echo_logger.log_interaction("SYSTEM_OUTPUT", no_feedback_msg)

    def provide_feedback(self):
        """Allow user to provide feedback on recent classifications."""
        print("Feedback functionality is integrated into the classification process.")
        print("When the model has low confidence (<80%), it will automatically ask for feedback.")
        print("You can also manually provide feedback by re-classifying messages.")

    def manage_models(self):
        """Model management interface."""
        while True:
            print("\n" + "-"*30)
            print("MODEL MANAGEMENT")
            print("-"*30)
            print("1. View existing model versions")
            print("2. Create new model version from original dataset")
            print("3. Create new model version from current dataset")
            print("4. Load a specific model version")
            print("5. Make a copy of existing model")
            print("6. Delete a model version")
            print("7. Back to main menu")

            choice = input("Enter your choice (1-7): ").strip()

            if choice == '1':
                self.view_models()
            elif choice == '2':
                self.create_model_from_original()
            elif choice == '3':
                self.create_model_from_current()
            elif choice == '4':
                self.load_model()
            elif choice == '5':
                self.copy_model()
            elif choice == '6':
                self.delete_model()
            elif choice == '7':
                break
            else:
                print("Invalid choice. Please try again.")

    def view_models(self):
        """View all available model versions."""
        print("\nAvailable Model Versions:")
        print("-" * 40)

        model_files = [f for f in os.listdir('.') if f.startswith(('SVM_', 'LogisticRegression_', 'RandomForest_')) and f.endswith('.pkl')]

        if not model_files:
            print("No model files found.")
            return

        current_model = None
        if os.path.exists('current_model.txt'):
            with open('current_model.txt', 'r') as f:
                current_model = f.read().strip()

        for i, model_file in enumerate(sorted(model_files), 1):
            marker = " <-- CURRENT" if model_file == current_model else ""
            print(f"{i}. {model_file}{marker}")

            # Try to load metadata
            metadata_file = model_file.replace('.pkl', '_metadata.pkl')
            if os.path.exists(metadata_file):
                try:
                    metadata = joblib.load(metadata_file)
                    print(f"   Created: {metadata.get('timestamp', 'Unknown')}")
                    metrics = metadata.get('metrics', {})
                    if metrics:
                        print(f"   Accuracy: {metrics.get('accuracy', 0):.3f}")
                except:
                    pass
            print()

    def create_model_from_original(self):
        """Create new model from original dataset."""
        print("Creating new model from original dataset...")
        print("This will run the training script on SMSSpamCollection.txt")

        confirm = input("This may take several minutes. Continue? (y/n): ").lower()
        if confirm != 'y':
            return

        try:
            # Run preprocessing and training
            os.system('python preprocess.py')
            os.system('python train_model.py')
            print("New model created successfully!")
            logger.log_model_action("MODEL_CREATED", "New model from original dataset")
            self.load_current_model()
        except Exception as e:
            print(f"Error creating model: {e}")
            logger.log_error("MODEL_CREATION_ERROR", str(e))

    def create_model_from_current(self):
        """Create new model from current dataset (with feedback)."""
        if not os.path.exists('user_feedback.log'):
            print("No feedback data available. Use 'Update' first to incorporate feedback.")
            return

        print("Creating new model from current dataset (including feedback)...")

        confirm = input("This may take several minutes. Continue? (y/n): ").lower()
        if confirm != 'y':
            return

        try:
            # Update dataset with feedback and retrain
            self.feedback_system.update_training_dataset()
            self.feedback_system.retrain_model()
            print("New model created with feedback data!")
            logger.log_model_action("MODEL_CREATED", "New model from current dataset with feedback")
            self.load_current_model()
        except Exception as e:
            print(f"Error creating model: {e}")
            logger.log_error("MODEL_CREATION_ERROR", str(e))

    def load_model(self):
        """Load a specific model version."""
        self.view_models()
        model_files = [f for f in os.listdir('.') if f.startswith(('SVM_', 'LogisticRegression_', 'RandomForest_')) and f.endswith('.pkl')]

        if not model_files:
            return

        try:
            choice = int(input("Enter the number of the model to load: ")) - 1
            if 0 <= choice < len(model_files):
                model_file = sorted(model_files)[choice]

                # Update current model reference
                with open('current_model.txt', 'w') as f:
                    f.write(model_file)

                # Reload classifier
                self.classifier = SMSClassifier(model_file)
                self.current_model = model_file

                print(f"Model {model_file} loaded successfully!")
                logger.log_model_action("MODEL_LOADED", f"Manually loaded {model_file}")
            else:
                print("Invalid choice.")
        except ValueError:
            print("Please enter a valid number.")

    def copy_model(self):
        """Make a copy of an existing model."""
        self.view_models()
        model_files = [f for f in os.listdir('.') if f.startswith(('SVM_', 'LogisticRegression_', 'RandomForest_')) and f.endswith('.pkl')]

        if not model_files:
            return

        try:
            choice = int(input("Enter the number of the model to copy: ")) - 1
            if 0 <= choice < len(model_files):
                source_file = sorted(model_files)[choice]
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                copy_file = f"{source_file.rsplit('.', 1)[0]}_copy_{timestamp}.pkl"

                # Copy the model file
                import shutil
                shutil.copy2(source_file, copy_file)

                # Copy metadata if it exists
                metadata_file = source_file.replace('.pkl', '_metadata.pkl')
                if os.path.exists(metadata_file):
                    copy_metadata = copy_file.replace('.pkl', '_metadata.pkl')
                    shutil.copy2(metadata_file, copy_metadata)

                print(f"Model copied as: {copy_file}")
                logger.log_model_action("MODEL_COPIED", f"{source_file} -> {copy_file}")
            else:
                print("Invalid choice.")
        except ValueError:
            print("Please enter a valid number.")
        except Exception as e:
            print(f"Error copying model: {e}")
            logger.log_error("MODEL_COPY_ERROR", str(e))

    def delete_model(self):
        """Delete a model version."""
        self.view_models()
        model_files = [f for f in os.listdir('.') if f.startswith(('SVM_', 'LogisticRegression_', 'RandomForest_')) and f.endswith('.pkl')]

        if not model_files:
            return

        try:
            choice = int(input("Enter the number of the model to delete: ")) - 1
            if 0 <= choice < len(model_files):
                model_file = sorted(model_files)[choice]

                # Check if it's the current model
                current_model = None
                if os.path.exists('current_model.txt'):
                    with open('current_model.txt', 'r') as f:
                        current_model = f.read().strip()

                if model_file == current_model:
                    print("Cannot delete the currently loaded model.")
                    return

                confirm = input(f"Are you sure you want to delete {model_file}? (y/n): ").lower()
                if confirm != 'y':
                    return

                # Delete model file and metadata
                os.remove(model_file)
                metadata_file = model_file.replace('.pkl', '_metadata.pkl')
                if os.path.exists(metadata_file):
                    os.remove(metadata_file)

                print(f"Model {model_file} deleted successfully!")
                logger.log_model_action("MODEL_DELETED", model_file)
            else:
                print("Invalid choice.")
        except ValueError:
            print("Please enter a valid number.")
        except Exception as e:
            print(f"Error deleting model: {e}")
            logger.log_error("MODEL_DELETE_ERROR", str(e))

    def update_dataset(self):
        """Update training dataset with user feedback."""
        if not os.path.exists('user_feedback.log'):
            print("No feedback data available.")
            return

        try:
            updated_file = self.feedback_system.update_training_dataset()
            print(f"Dataset updated successfully: {updated_file}")
            logger.log_model_action("DATASET_UPDATED", f"Feedback incorporated into {updated_file}")
        except Exception as e:
            print(f"Error updating dataset: {e}")
            logger.log_error("DATASET_UPDATE_ERROR", str(e))

    def retrain_model(self):
        """Retrain the current model."""
        if not os.path.exists('SMSSpamCollection_updated.txt'):
            print("No updated dataset available. Use 'Update' first.")
            return

        print("Retraining model on updated dataset...")

        confirm = input("This may take several minutes. Continue? (y/n): ").lower()
        if confirm != 'y':
            return

        try:
            model_file = self.feedback_system.retrain_model()
            print(f"Model retrained successfully: {model_file}")
            logger.log_model_action("MODEL_RETRAINED", model_file)
            self.load_current_model()
        except Exception as e:
            print(f"Error retraining model: {e}")
            logger.log_error("MODEL_RETRAIN_ERROR", str(e))

    def show_stats(self):
        """Display last 5 log entries from each log file."""
        print("\n" + "="*60)
        print("LOG STATISTICS - Last 5 entries from each log")
        print("="*60)

        log_types = ['model_actions', 'user_feedback', 'error_reports', 'echo_interactions']

        for log_type in log_types:
            print(f"\n{log_type.upper().replace('_', ' ')}:")
            print("-" * 40)

            if log_type == 'echo_interactions':
                # Handle echo interactions log separately
                echo_log_path = os.path.join('.', 'echo_interactions.log')
                if os.path.exists(echo_log_path):
                    try:
                        with open(echo_log_path, 'r', encoding='utf-8') as f:
                            lines = f.readlines()
                        recent_logs = lines[-5:] if len(lines) >= 5 else lines
                        for i, log_entry in enumerate(recent_logs, 1):
                            print(f"{i}. {log_entry.strip()}")
                    except Exception as e:
                        print(f"Error reading echo log: {e}")
                else:
                    print("No entries found.")
            else:
                recent_logs = logger.get_recent_logs(log_type, 5)

                if not recent_logs or recent_logs[0].startswith("Log file does not exist"):
                    print("No entries found.")
                else:
                    for i, log_entry in enumerate(recent_logs, 1):
                        if log_type == 'error_reports':
                            # Extract ERROR_TYPE and timestamp
                            # Format: "2023-10-19 10:30:45,123 - Error Reports - ERROR - ERROR_TYPE: TYPE - MESSAGE: msg"
                            parts = log_entry.split(' - ')
                            if len(parts) >= 4:
                                timestamp = parts[0]
                                error_part = ' - '.join(parts[3:])
                                if 'ERROR_TYPE:' in error_part:
                                    error_type = error_part.split('ERROR_TYPE:')[1].split(' - ')[0].strip()
                                    print(f"{i}. ERROR_TYPE: {error_type} - TIMESTAMP: {timestamp}")
                                else:
                                    print(f"{i}. {log_entry}")
                            else:
                                print(f"{i}. {log_entry}")
                        elif log_type == 'user_feedback':
                            # Extract prediction type, correctness, and timestamp
                            # Format: "2023-10-19 10:30:45,123 - User Feedback - INFO - FEEDBACK: {"message": "...", "predicted": "spam", "correct": "ham", "confidence": 0.95, "timestamp": "..."}"
                            parts = log_entry.split(' - ')
                            if len(parts) >= 4:
                                timestamp = parts[0]
                                feedback_json = ' - '.join(parts[3:]).replace('FEEDBACK: ', '')
                                try:
                                    feedback_data = json.loads(feedback_json)
                                    predicted = feedback_data.get('predicted', 'unknown').upper()
                                    correct = feedback_data.get('correct', 'unknown').upper()
                                    is_correct = "CORRECT" if predicted == correct else "INCORRECT"
                                    print(f"{i}. PREDICTION: {predicted} - CORRECTNESS: {is_correct} - TIMESTAMP: {timestamp}")
                                except json.JSONDecodeError:
                                    print(f"{i}. {log_entry}")
                            else:
                                print(f"{i}. {log_entry}")
                        elif log_type == 'model_actions':
                            # Extract action type and timestamp
                            # Format: "2023-10-19 10:30:45,123 - Model Actions - INFO - ACTION: TYPE - DETAILS: ..."
                            parts = log_entry.split(' - ')
                            if len(parts) >= 4:
                                timestamp = parts[0]
                                action_part = ' - '.join(parts[3:])
                                if 'ACTION:' in action_part:
                                    action_type = action_part.split('ACTION:')[1].split(' - ')[0].strip()
                                    print(f"{i}. ACTION: {action_type} - TIMESTAMP: {timestamp}")
                                else:
                                    print(f"{i}. {log_entry}")
                            else:
                                print(f"{i}. {log_entry}")
                        else:
                            # Fallback for any other log types
                            print(f"{i}. {log_entry}")

    def benchmark_model(self):
        """Run performance benchmarking on the current model."""
        if not self.classifier.model:
            print("No model loaded for benchmarking.")
            return

        print("Running performance benchmark...")
        print("This will test the model on sample messages and measure timing.")

        # Sample messages for benchmarking
        test_messages = [
            "Hey, how are you doing?",
            "WINNER! You have won a £1000 cash prize! Call now!",
            "Meeting scheduled for tomorrow at 3 PM",
            "URGENT: Your account needs verification. Click this link immediately!",
            "Thanks for your help with the project",
            "FREE entry into our competition. Text YES to enter!",
            "Hi mom, I'll be home late tonight",
            "Congratulations! You've been selected for a special offer!",
            "Can we reschedule our call?",
            "ALERT: Suspicious activity detected on your account"
        ]

        start_time = time.time()
        results = []

        for msg in test_messages:
            msg_start = time.time()
            result = self.classifier.classify_message(msg)
            msg_end = time.time()

            if "error" not in result:
                results.append({
                    'message': msg,
                    'classification': result['classification'],
                    'confidence': result['confidence'],
                    'time': msg_end - msg_start
                })

        total_time = time.time() - start_time

        # Calculate statistics
        if results:
            avg_time = sum(r['time'] for r in results) / len(results)
            avg_confidence = sum(r['confidence'] for r in results) / len(results)
            spam_count = sum(1 for r in results if r['classification'] == 'spam')

            print("\nBenchmark Results:")
            print(f"Average classification time: {avg_time:.3f} seconds")
            print(f"Average confidence: {avg_confidence:.3f}")
            print(f"Total benchmark time: {total_time:.3f} seconds")
            print(f"Spam detections: {spam_count}/{len(results)}")
            print(f"Total benchmark time: {total_time:.1f} seconds")

            # Memory usage estimate (rough)
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            print(f"Memory usage: {memory_mb:.1f} MB")

            logger.log_model_action("BENCHMARK_COMPLETED",
                                   f"Average time: {avg_time:.3f}s, Avg confidence: {avg_confidence:.3f}, Total time: {total_time:.3f}s")
        else:
            print("Benchmark failed - no valid results.")

    def show_help(self):
        """Show help information."""
        print("\n" + "="*60)
        print("SMS SPAM DETECTION SYSTEM - HELP")
        print("="*60)
        print("1. Classify - Enter any SMS message to get spam/ham classification")
        print("   with confidence score. Low confidence prompts for feedback.")
        print()
        print("2. Feedback - Manual feedback is handled during classification.")
        print("   The system automatically requests feedback for uncertain predictions.")
        print()
        print("3. Manage Models - Complete model lifecycle management:")
        print("   - View all saved model versions with metadata")
        print("   - Create new models from original or current dataset")
        print("   - Load specific model versions")
        print("   - Copy existing models for backup/experimentation")
        print("   - Delete unwanted model versions")
        print()
        print("4. Update - Incorporates user feedback into the training dataset")
        print("   for continuous model improvement.")
        print()
        print("5. Retrain - Retrains the model using the updated dataset")
        print("   with accumulated feedback data.")
        print()
        print("6. Stats - Shows recent activity from all three log files:")
        print("   - model_actions.log: Training, loading, classification actions")
        print("   - user_feedback.log: All user feedback provided")
        print("   - error_reports.log: System errors and issues")
        print()
        print("7. Benchmark - Performance testing on current model:")
        print("   - Average classification time")
        print("   - Memory usage")
        print("   - Accuracy metrics on test set")
        print()
        print("8. Help - This help screen")
        print()
        print("9. Quit - Saves current model state and exits cleanly")
        print("="*60)

    def quit_system(self):
        """Quit the system and save current model."""
        print("Saving current model state...")

        if self.current_model:
            logger.log_model_action("SYSTEM_SHUTDOWN", f"Current model: {self.current_model}")
        else:
            logger.log_model_action("SYSTEM_SHUTDOWN", "No model loaded")

        # Auto-disable echo logging on quit
        if echo_logger.echo_enabled:
            echo_logger.end_session()
            print("Echo logging has been automatically disabled.")

        # Cleanup old logs
        logger.cleanup_old_logs()

        print("Thank you for using SMS Spam Detection System!")
        print("System shutdown complete.")
        sys.exit(0)

    def run(self):
        """Main application loop."""
        print("Welcome to SMS Spam Detection System!")
        print("Type 'help' or select option 8 for command information.")

        while True:
            self.display_menu()
            choice = input("Enter your choice (1-9): ").strip()

            # Log user input if echo logging is enabled
            echo_logger.log_interaction("USER_INPUT", choice)

            if choice == '1':
                self.classify_message()
            elif choice == '2':
                self.provide_feedback()
            elif choice == '3':
                self.manage_models()
            elif choice == '4':
                self.update_dataset()
            elif choice == '5':
                self.retrain_model()
            elif choice == '6':
                self.show_stats()
            elif choice == '7':
                self.benchmark_model()
            elif choice == '8' or choice.lower() == 'help':
                self.show_help()
            elif choice == '9' or choice.lower() == 'quit':
                self.quit_system()
            else:
                print("Invalid choice. Please enter a number between 1-9.")

if __name__ == "__main__":
    app = SMSSpamDetector()
    app.run()