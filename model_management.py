import os
import shutil
from datetime import datetime
import joblib
from inference_engine import SMSClassifier
from logging_system import logger

class ModelManager:
    """Comprehensive model management system for SMS spam detection."""

    def __init__(self):
        self.models_dir = "models"
        self.current_model_file = "current_model.txt"
        os.makedirs(self.models_dir, exist_ok=True)

    def list_models(self):
        """List all available model versions with metadata."""
        model_files = [f for f in os.listdir('.') if f.startswith(('SVM_', 'LogisticRegression_', 'RandomForest_')) and f.endswith('.pkl')]

        if not model_files:
            print("No model files found.")
            return []

        current_model = self._get_current_model()

        models_info = []
        for i, model_file in enumerate(sorted(model_files), 1):
            is_current = model_file == current_model
            model_info = {
                "index": i,
                "filename": model_file,
                "is_current": is_current,
                "size_mb": os.path.getsize(model_file) / (1024 * 1024)
            }

            # Load metadata if available
            metadata_file = model_file.replace('.pkl', '_metadata.pkl')
            if os.path.exists(metadata_file):
                try:
                    metadata = joblib.load(metadata_file)
                    model_info.update({
                        "model_name": metadata.get('model_name', 'Unknown'),
                        "created": metadata.get('timestamp', 'Unknown'),
                        "accuracy": metadata.get('metrics', {}).get('accuracy', 'N/A'),
                        "f1_score": metadata.get('metrics', {}).get('f1', 'N/A')
                    })
                except Exception as e:
                    model_info["metadata_error"] = str(e)

            models_info.append(model_info)

        return models_info

    def display_models(self):
        """Display formatted list of models."""
        models = self.list_models()

        if not models:
            return

        print("\n" + "="*80)
        print("AVAILABLE MODEL VERSIONS")
        print("="*80)
        print("<10")
        print("-" * 80)

        for model in models:
            current_marker = " <-- CURRENT" if model["is_current"] else ""
            print("<2")
            print("<15")

            if "model_name" in model:
                print("<12")
                print("<10")
                print("<10")

            print("-" * 80)

    def create_model_from_original(self):
        """Create new model from original dataset."""
        print("Creating new model from original dataset...")
        print("This will run preprocessing and training on SMSSpamCollection.txt")

        confirm = input("This may take several minutes. Continue? (y/n): ").lower()
        if confirm != 'y':
            return False

        try:
            # Run preprocessing and training
            os.system('python preprocess.py')
            os.system('python train_model.py')

            # Move model files to models directory for organization
            self._organize_model_files()

            logger.log_model_action("MODEL_CREATED", "New model from original dataset")
            print("New model created successfully!")
            return True

        except Exception as e:
            logger.log_error("MODEL_CREATION_ERROR", str(e))
            print(f"Error creating model: {e}")
            return False

    def create_model_from_current(self):
        """Create new model from current dataset with feedback."""
        if not os.path.exists('user_feedback.log'):
            print("No feedback data available. Use 'Update Dataset' first.")
            return False

        print("Creating new model from current dataset (including feedback)...")

        confirm = input("This may take several minutes. Continue? (y/n): ").lower()
        if confirm != 'y':
            return False

        try:
            # Update dataset with feedback and retrain
            from feedback_system import FeedbackSystem
            feedback_sys = FeedbackSystem()
            feedback_sys.update_training_dataset()
            feedback_sys.retrain_model()

            # Organize model files
            self._organize_model_files()

            logger.log_model_action("MODEL_CREATED", "New model from current dataset with feedback")
            print("New model created with feedback data!")
            return True

        except Exception as e:
            logger.log_error("MODEL_CREATION_ERROR", str(e))
            print(f"Error creating model: {e}")
            return False

    def load_model(self, model_index=None):
        """Load a specific model version."""
        models = self.list_models()

        if not models:
            print("No models available to load.")
            return False

        if model_index is None:
            self.display_models()
            try:
                choice = int(input("Enter the number of the model to load: ")) - 1
            except ValueError:
                print("Please enter a valid number.")
                return False
        else:
            choice = model_index - 1

        if 0 <= choice < len(models):
            model_file = models[choice]["filename"]

            # Update current model reference
            with open(self.current_model_file, 'w') as f:
                f.write(model_file)

            print(f"Model {model_file} loaded successfully!")
            logger.log_model_action("MODEL_LOADED", f"Manually loaded {model_file}")
            return True
        else:
            print("Invalid choice.")
            return False

    def copy_model(self, model_index=None):
        """Make a copy of an existing model."""
        models = self.list_models()

        if not models:
            print("No models available to copy.")
            return False

        if model_index is None:
            self.display_models()
            try:
                choice = int(input("Enter the number of the model to copy: ")) - 1
            except ValueError:
                print("Please enter a valid number.")
                return False
        else:
            choice = model_index - 1

        if 0 <= choice < len(models):
            source_file = models[choice]["filename"]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            copy_file = f"{source_file.rsplit('.', 1)[0]}_copy_{timestamp}.pkl"

            try:
                # Copy model file
                shutil.copy2(source_file, copy_file)

                # Copy metadata if it exists
                metadata_file = source_file.replace('.pkl', '_metadata.pkl')
                if os.path.exists(metadata_file):
                    copy_metadata = copy_file.replace('.pkl', '_metadata.pkl')
                    shutil.copy2(metadata_file, copy_metadata)

                # Move to models directory
                self._move_to_models_dir(copy_file)
                if os.path.exists(copy_metadata):
                    self._move_to_models_dir(copy_metadata)

                print(f"Model copied as: {copy_file}")
                logger.log_model_action("MODEL_COPIED", f"{source_file} -> {copy_file}")
                return True

            except Exception as e:
                logger.log_error("MODEL_COPY_ERROR", str(e))
                print(f"Error copying model: {e}")
                return False
        else:
            print("Invalid choice.")
            return False

    def delete_model(self, model_index=None):
        """Delete a model version."""
        models = self.list_models()

        if not models:
            print("No models available to delete.")
            return False

        if model_index is None:
            self.display_models()
            try:
                choice = int(input("Enter the number of the model to delete: ")) - 1
            except ValueError:
                print("Please enter a valid number.")
                return False
        else:
            choice = model_index - 1

        if 0 <= choice < len(models):
            model_file = models[choice]["filename"]

            # Check if it's the current model
            if models[choice]["is_current"]:
                print("Cannot delete the currently loaded model.")
                print("Please load a different model first.")
                return False

            confirm = input(f"Are you sure you want to delete {model_file}? (y/n): ").lower()
            if confirm != 'y':
                return False

            try:
                # Delete model file and metadata
                os.remove(model_file)
                metadata_file = model_file.replace('.pkl', '_metadata.pkl')
                if os.path.exists(metadata_file):
                    os.remove(metadata_file)

                print(f"Model {model_file} deleted successfully!")
                logger.log_model_action("MODEL_DELETED", model_file)
                return True

            except Exception as e:
                logger.log_error("MODEL_DELETE_ERROR", str(e))
                print(f"Error deleting model: {e}")
                return False
        else:
            print("Invalid choice.")
            return False

    def get_model_stats(self):
        """Get statistics about all models."""
        models = self.list_models()

        if not models:
            return {"total_models": 0, "total_size_mb": 0, "current_model": None}

        total_size = sum(model["size_mb"] for model in models)
        current_model = next((m for m in models if m["is_current"]), None)

        stats = {
            "total_models": len(models),
            "total_size_mb": total_size,
            "current_model": current_model["filename"] if current_model else None,
            "models_by_type": {}
        }

        # Group by model type
        for model in models:
            model_type = model["filename"].split('_')[0]
            if model_type not in stats["models_by_type"]:
                stats["models_by_type"][model_type] = 0
            stats["models_by_type"][model_type] += 1

        return stats

    def cleanup_old_models(self, keep_recent=5):
        """Keep only the most recent N models of each type."""
        models = self.list_models()

        if len(models) <= keep_recent:
            print("No old models to clean up.")
            return 0

        # Group models by type
        models_by_type = {}
        for model in models:
            model_type = model["filename"].split('_')[0]
            if model_type not in models_by_type:
                models_by_type[model_type] = []
            models_by_type[model_type].append(model)

        deleted_count = 0

        for model_type, type_models in models_by_type.items():
            if len(type_models) > keep_recent:
                # Sort by creation date (filename contains timestamp)
                type_models.sort(key=lambda x: x["filename"], reverse=True)

                # Keep the most recent, delete the rest
                to_delete = type_models[keep_recent:]

                for model in to_delete:
                    if not model["is_current"]:  # Don't delete current model
                        try:
                            os.remove(model["filename"])
                            metadata_file = model["filename"].replace('.pkl', '_metadata.pkl')
                            if os.path.exists(metadata_file):
                                os.remove(metadata_file)

                            logger.log_model_action("MODEL_AUTO_DELETED", f"Old model cleanup: {model['filename']}")
                            deleted_count += 1
                        except Exception as e:
                            logger.log_error("MODEL_CLEANUP_ERROR", f"Failed to delete {model['filename']}: {str(e)}")

        if deleted_count > 0:
            print(f"Cleaned up {deleted_count} old model(s).")
        else:
            print("No old models were deleted (kept current model and recent versions).")

        return deleted_count

    def _get_current_model(self):
        """Get the filename of the currently loaded model."""
        if os.path.exists(self.current_model_file):
            with open(self.current_model_file, 'r') as f:
                return f.read().strip()
        return None

    def _organize_model_files(self):
        """Move model files to organized directory structure."""
        # This is optional - for now, keep models in root directory
        # But could be extended to move to models/ subdirectories by type
        pass

    def _move_to_models_dir(self, filename):
        """Move a file to the models directory."""
        if not os.path.exists(filename):
            return

        dest_path = os.path.join(self.models_dir, os.path.basename(filename))
        shutil.move(filename, dest_path)
        return dest_path

def main():
    """Command-line interface for model management."""
    manager = ModelManager()

    while True:
        print("\n" + "="*50)
        print("MODEL MANAGEMENT SYSTEM")
        print("="*50)
        print("1. List all models")
        print("2. Create model from original dataset")
        print("3. Create model from current dataset")
        print("4. Load a model")
        print("5. Copy a model")
        print("6. Delete a model")
        print("7. Show model statistics")
        print("8. Cleanup old models")
        print("9. Back to main menu")

        choice = input("Enter your choice (1-9): ").strip()

        if choice == '1':
            manager.display_models()
        elif choice == '2':
            manager.create_model_from_original()
        elif choice == '3':
            manager.create_model_from_current()
        elif choice == '4':
            manager.load_model()
        elif choice == '5':
            manager.copy_model()
        elif choice == '6':
            manager.delete_model()
        elif choice == '7':
            stats = manager.get_model_stats()
            print("\nModel Statistics:")
            print(f"Total models: {stats['total_models']}")
            print(".1f")
            print(f"Current model: {stats['current_model'] or 'None'}")
            print("Models by type:")
            for model_type, count in stats['models_by_type'].items():
                print(f"  {model_type}: {count}")
        elif choice == '8':
            keep_recent = input("How many recent models to keep per type? (default 5): ").strip()
            keep_recent = int(keep_recent) if keep_recent.isdigit() else 5
            manager.cleanup_old_models(keep_recent)
        elif choice == '9':
            break
        else:
            print("Invalid choice. Please enter a number between 1-9.")

if __name__ == "__main__":
    main()