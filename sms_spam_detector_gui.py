#!/usr/bin/env python3
"""
SMS Spam Detection GUI - Tkinter Interface
A comprehensive AI-powered SMS spam detection system with graphical user interface.
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import threading
import time
import os
from datetime import datetime
from inference_engine import SMSClassifier
from feedback_system import FeedbackSystem
from logging_system import logger, echo_logger
import joblib
import json

class SMSSpamDetectorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("SMS Spam Detection System")
        self.root.geometry("800x600")

        # Status bar (create early for update_status to work)
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        self.status_bar = ttk.Label(root, textvariable=self.status_var, relief=tk.SUNKEN)
        self.status_bar.pack(fill='x', side='bottom')

        # Initialize backend components
        self.classifier = SMSClassifier()
        self.feedback_system = FeedbackSystem()
        self.current_model = None
        self.load_current_model()

        # Setup echo logging
        self.setup_echo_logging()

        # Create main tabbed interface
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)

        # Create tabs
        self.create_classify_tab()
        self.create_model_management_tab()
        self.create_dataset_tab()
        self.create_retrain_tab()
        self.create_stats_tab()
        self.create_benchmark_tab()
        self.create_help_tab()

        # Initialize feedback dialog state
        self.feedback_pending = False
        self.pending_feedback_data = None

    def load_current_model(self):
        """Load the current model on startup."""
        try:
            if os.path.exists('current_model.txt'):
                with open('current_model.txt', 'r') as f:
                    model_file = f.read().strip()
                self.current_model = model_file
                self.update_status(f"Loaded model: {model_file}")
                logger.log_model_action("MODEL_LOADED", f"Loaded {model_file} on GUI startup")
            else:
                self.update_status("No saved model found. Please train a model first.")
                logger.log_error("NO_MODEL_FOUND", "No current_model.txt file found on GUI startup")
        except Exception as e:
            self.update_status(f"Error loading model: {e}")
            logger.log_error("MODEL_LOAD_ERROR", f"Failed to load model on GUI startup: {str(e)}")

    def setup_echo_logging(self):
        """Set up echo logging and prompt user on startup."""
        # For GUI, we'll enable echo logging by default or provide a toggle
        # For now, disable it to avoid cluttering the interface
        if echo_logger.echo_enabled:
            echo_logger.disable_echo_log()

    def update_status(self, message):
        """Update the status bar message."""
        self.status_var.set(message)
        self.root.update_idletasks()

    def create_classify_tab(self):
        """Create the classification tab."""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Classify")

        # Message input
        ttk.Label(frame, text="Enter SMS Message:").pack(pady=(10, 5))
        self.message_text = scrolledtext.ScrolledText(frame, height=5, wrap=tk.WORD)
        self.message_text.pack(fill='x', padx=10, pady=(0, 10))

        # Classify button
        self.classify_btn = ttk.Button(frame, text="Classify Message", command=self.classify_message)
        self.classify_btn.pack(pady=(0, 10))

        # Results display
        ttk.Label(frame, text="Classification Result:").pack(pady=(10, 5))
        self.result_text = tk.Text(frame, height=8, wrap=tk.WORD, state='disabled')
        self.result_text.pack(fill='x', padx=10, pady=(0, 10))

        # Feedback section (initially hidden)
        self.feedback_frame = ttk.LabelFrame(frame, text="Feedback Required")
        self.feedback_label = ttk.Label(self.feedback_frame, text="")
        self.feedback_label.pack(pady=10)

        feedback_btn_frame = ttk.Frame(self.feedback_frame)
        feedback_btn_frame.pack(pady=(0, 10))

        self.feedback_yes_btn = ttk.Button(feedback_btn_frame, text="Yes", command=lambda: self.submit_feedback("Yes"))
        self.feedback_yes_btn.pack(side='left', padx=(0, 5))

        self.feedback_no_btn = ttk.Button(feedback_btn_frame, text="No", command=lambda: self.submit_feedback("No"))
        self.feedback_no_btn.pack(side='left', padx=(0, 5))

        self.feedback_skip_btn = ttk.Button(feedback_btn_frame, text="Skip", command=lambda: self.submit_feedback("Skip"))
        self.feedback_skip_btn.pack(side='left')

    def create_model_management_tab(self):
        """Create the model management tab."""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Model Management")

        # Model list
        ttk.Label(frame, text="Available Models:").pack(pady=(10, 5))
        self.model_listbox = tk.Listbox(frame, height=10)
        self.model_listbox.pack(fill='both', expand=True, padx=10, pady=(0, 10))

        # Buttons
        btn_frame = ttk.Frame(frame)
        btn_frame.pack(fill='x', padx=10, pady=(0, 10))

        ttk.Button(btn_frame, text="Refresh List", command=self.refresh_model_list).pack(side='left', padx=(0, 5))
        ttk.Button(btn_frame, text="View Details", command=self.view_model_details).pack(side='left', padx=(0, 5))
        ttk.Button(btn_frame, text="Load Model", command=self.load_selected_model).pack(side='left', padx=(0, 5))
        ttk.Button(btn_frame, text="Copy Model", command=self.copy_selected_model).pack(side='left', padx=(0, 5))
        ttk.Button(btn_frame, text="Delete Model", command=self.delete_selected_model).pack(side='left')

        # Create new model section
        create_frame = ttk.LabelFrame(frame, text="Create New Model")
        create_frame.pack(fill='x', padx=10, pady=(10, 0))

        ttk.Button(create_frame, text="From Original Dataset", command=self.create_model_from_original).pack(side='left', padx=(10, 5))
        ttk.Button(create_frame, text="From Current Dataset", command=self.create_model_from_current).pack(side='left')

    def create_dataset_tab(self):
        """Create the dataset update tab."""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Update Dataset")

        ttk.Label(frame, text="Update training dataset with accumulated user feedback.").pack(pady=(20, 10))

        self.update_btn = ttk.Button(frame, text="Update Dataset with Feedback", command=self.update_dataset)
        self.update_btn.pack(pady=(0, 20))

        self.dataset_status = tk.Text(frame, height=10, wrap=tk.WORD, state='disabled')
        self.dataset_status.pack(fill='both', expand=True, padx=10, pady=10)

    def create_retrain_tab(self):
        """Create the retrain tab."""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Retrain Model")

        ttk.Label(frame, text="Retrain the model using the updated dataset.").pack(pady=(20, 10))

        self.retrain_btn = ttk.Button(frame, text="Retrain Model", command=self.retrain_model)
        self.retrain_btn.pack(pady=(0, 10))

        # Progress bar
        self.retrain_progress = ttk.Progressbar(frame, orient='horizontal', mode='indeterminate')
        self.retrain_progress.pack(fill='x', padx=10, pady=(0, 10))

        self.retrain_status = tk.Text(frame, height=10, wrap=tk.WORD, state='disabled')
        self.retrain_status.pack(fill='both', expand=True, padx=10, pady=10)

    def create_stats_tab(self):
        """Create the statistics tab."""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Statistics")

        # Log selection
        ttk.Label(frame, text="Select Log Type:").pack(pady=(10, 5))
        self.log_type_var = tk.StringVar(value="model_actions")
        log_combo = ttk.Combobox(frame, textvariable=self.log_type_var,
                                values=["model_actions", "user_feedback", "error_reports", "echo_interactions"])
        log_combo.pack(pady=(0, 10))
        log_combo.bind('<<ComboboxSelected>>', self.update_stats_display)

        # Stats display
        self.stats_text = scrolledtext.ScrolledText(frame, wrap=tk.WORD)
        self.stats_text.pack(fill='both', expand=True, padx=10, pady=(0, 10))

        ttk.Button(frame, text="Refresh Stats", command=self.update_stats_display).pack(pady=(0, 10))

    def create_benchmark_tab(self):
        """Create the benchmark tab."""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Benchmark")

        ttk.Label(frame, text="Run performance benchmarking on the current model.").pack(pady=(20, 10))

        self.benchmark_btn = ttk.Button(frame, text="Run Benchmark", command=self.run_benchmark)
        self.benchmark_btn.pack(pady=(0, 10))

        # Progress bar
        self.benchmark_progress = ttk.Progressbar(frame, orient='horizontal', mode='indeterminate')
        self.benchmark_progress.pack(fill='x', padx=10, pady=(0, 10))

        self.benchmark_results = tk.Text(frame, height=15, wrap=tk.WORD, state='disabled')
        self.benchmark_results.pack(fill='both', expand=True, padx=10, pady=10)

    def create_help_tab(self):
        """Create the help tab."""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Help")

        help_text = scrolledtext.ScrolledText(frame, wrap=tk.WORD)
        help_text.pack(fill='both', expand=True, padx=10, pady=10)

        help_content = """
SMS SPAM DETECTION SYSTEM - HELP

CLASSIFY TAB:
- Enter any SMS message in the text area
- Click "Classify Message" to get spam/ham classification with confidence score
- If confidence is low (<80%), you'll be prompted for feedback to improve the model

MODEL MANAGEMENT TAB:
- View all available model versions with metadata
- Create new models from original or current dataset (with feedback)
- Load specific model versions
- Copy existing models for backup
- Delete unwanted model versions

UPDATE DATASET TAB:
- Incorporates accumulated user feedback into the training dataset
- Prepares data for model retraining

RETRAIN MODEL TAB:
- Retrains the model using the updated dataset with feedback data
- Shows progress during training

STATISTICS TAB:
- Shows recent activity from all log files:
  - model_actions.log: Training, loading, classification actions
  - user_feedback.log: All user feedback provided
  - error_reports.log: System errors and issues
  - echo_interactions.log: User interactions (if enabled)

BENCHMARK TAB:
- Performance testing on current model
- Measures average classification time, memory usage, and accuracy

HELP TAB:
- This help screen

The system automatically saves the current model state when closed.
        """

        help_text.insert('1.0', help_content)
        help_text.config(state='disabled')

    # Tab implementation methods will be added next

    def classify_message(self):
        """Classify the message from the input field."""
        message = self.message_text.get('1.0', tk.END).strip()
        if not message:
            messagebox.showerror("Error", "Please enter a message to classify.")
            return

        if not self.classifier.model:
            messagebox.showerror("Error", "No model loaded. Please load or train a model first.")
            return

        self.update_status("Classifying message...")
        logger.log_model_action("CLASSIFICATION_REQUEST", f"Message length: {len(message)}")

        result = self.classifier.classify_message(message)

        if "error" in result:
            self.update_status(f"Classification error: {result['error']}")
            logger.log_error("CLASSIFICATION_ERROR", result['error'])
            messagebox.showerror("Classification Error", result['error'])
            return

        # Display result
        result_text = f"Classification: {result['classification'].upper()}\n"
        result_text += f"Confidence: {result['confidence']:.3f}\n"
        result_text += f"Ham Probability: {result['probabilities']['ham']:.3f}\n"
        result_text += f"Spam Probability: {result['probabilities']['spam']:.3f}"

        self.result_text.config(state='normal')
        self.result_text.delete('1.0', tk.END)
        self.result_text.insert('1.0', result_text)
        self.result_text.config(state='disabled')

        logger.log_model_action("MESSAGE_CLASSIFIED",
                               f"Result: {result['classification']}, Confidence: {result['confidence']:.3f}")

        # Handle feedback if needed
        if result['needs_feedback']:
            self.show_feedback_prompt(message, result['classification'], result['confidence'])
        else:
            self.hide_feedback_prompt()
            self.update_status("Classification complete - high confidence")

    def show_feedback_prompt(self, message, predicted, confidence):
        """Show feedback prompt for low confidence classifications."""
        self.feedback_pending = True
        self.pending_feedback_data = (message, predicted, confidence)

        feedback_data = self.feedback_system.get_feedback_prompt(message, predicted, confidence)
        self.feedback_label.config(text=feedback_data['prompt_text'])
        self.feedback_frame.pack(fill='x', padx=10, pady=(10, 0))

    def hide_feedback_prompt(self):
        """Hide the feedback prompt."""
        self.feedback_pending = False
        self.pending_feedback_data = None
        self.feedback_frame.pack_forget()

    def submit_feedback(self, response):
        """Submit user feedback."""
        if not self.feedback_pending or not self.pending_feedback_data:
            return

        message, predicted, confidence = self.pending_feedback_data

        try:
            result = self.feedback_system.submit_feedback(message, predicted, confidence, response)
            if result is True:
                self.update_status("Feedback submitted - prediction was correct")
            elif result is False:
                self.update_status("Feedback submitted - prediction was incorrect")
            else:
                self.update_status("Feedback skipped")
        except Exception as e:
            messagebox.showerror("Feedback Error", f"Error submitting feedback: {e}")

        self.hide_feedback_prompt()

    # Model management methods
    def refresh_model_list(self):
        """Refresh the model list display."""
        self.model_listbox.delete(0, tk.END)

        model_files = [f for f in os.listdir('.') if f.startswith(('SVM_', 'LogisticRegression_', 'RandomForest_')) and f.endswith('.pkl')]

        if not model_files:
            self.model_listbox.insert(tk.END, "No model files found.")
            return

        current_model = None
        if os.path.exists('current_model.txt'):
            with open('current_model.txt', 'r') as f:
                current_model = f.read().strip()

        for model_file in sorted(model_files):
            marker = " <-- CURRENT" if model_file == current_model else ""
            self.model_listbox.insert(tk.END, f"{model_file}{marker}")

            # Try to load metadata
            metadata_file = model_file.replace('.pkl', '_metadata.pkl')
            if os.path.exists(metadata_file):
                try:
                    metadata = joblib.load(metadata_file)
                    created = metadata.get('timestamp', 'Unknown')
                    accuracy = metadata.get('metrics', {}).get('accuracy', 0)
                    self.model_listbox.insert(tk.END, f"  Created: {created}, Accuracy: {accuracy:.3f}")
                except:
                    pass
            self.model_listbox.insert(tk.END, "")  # Empty line

    def view_model_details(self):
        """View details of selected model."""
        selection = self.model_listbox.curselection()
        if not selection:
            messagebox.showwarning("Selection Error", "Please select a model first.")
            return

        model_line = self.model_listbox.get(selection[0])
        if "--> CURRENT" in model_line or "Created:" in model_line or model_line.strip() == "":
            messagebox.showwarning("Selection Error", "Please select a model file name.")
            return

        model_file = model_line.split(" <--")[0]

        # Get model info
        self.classifier.model_path = model_file
        info = self.classifier.get_model_info()

        if "error" in info:
            messagebox.showerror("Model Info Error", info['error'])
            return

        details = f"Model File: {info['model_file']}\n"
        details += f"Model Name: {info['model_name']}\n"
        details += f"Created: {info['timestamp']}\n"
        if info['metrics']:
            details += f"Metrics:\n"
            for key, value in info['metrics'].items():
                details += f"  {key}: {value}\n"

        messagebox.showinfo("Model Details", details)

    def load_selected_model(self):
        """Load the selected model."""
        selection = self.model_listbox.curselection()
        if not selection:
            messagebox.showwarning("Selection Error", "Please select a model first.")
            return

        model_line = self.model_listbox.get(selection[0])
        if "--> CURRENT" in model_line or "Created:" in model_line or model_line.strip() == "":
            messagebox.showwarning("Selection Error", "Please select a model file name.")
            return

        model_file = model_line.split(" <--")[0]

        # Update current model reference
        with open('current_model.txt', 'w') as f:
            f.write(model_file)

        # Reload classifier
        self.classifier = SMSClassifier(model_file)
        self.current_model = model_file

        self.update_status(f"Model {model_file} loaded successfully!")
        logger.log_model_action("MODEL_LOADED", f"Manually loaded {model_file}")
        self.refresh_model_list()

    def copy_selected_model(self):
        """Copy the selected model."""
        selection = self.model_listbox.curselection()
        if not selection:
            messagebox.showwarning("Selection Error", "Please select a model first.")
            return

        model_line = self.model_listbox.get(selection[0])
        if "--> CURRENT" in model_line or "Created:" in model_line or model_line.strip() == "":
            messagebox.showwarning("Selection Error", "Please select a model file name.")
            return

        source_file = model_line.split(" <--")[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        copy_file = f"{source_file.rsplit('.', 1)[0]}_copy_{timestamp}.pkl"

        try:
            import shutil
            shutil.copy2(source_file, copy_file)

            # Copy metadata if it exists
            metadata_file = source_file.replace('.pkl', '_metadata.pkl')
            if os.path.exists(metadata_file):
                copy_metadata = copy_file.replace('.pkl', '_metadata.pkl')
                shutil.copy2(metadata_file, copy_metadata)

            self.update_status(f"Model copied as: {copy_file}")
            logger.log_model_action("MODEL_COPIED", f"{source_file} -> {copy_file}")
            self.refresh_model_list()
        except Exception as e:
            messagebox.showerror("Copy Error", f"Error copying model: {e}")
            logger.log_error("MODEL_COPY_ERROR", str(e))

    def delete_selected_model(self):
        """Delete the selected model."""
        selection = self.model_listbox.curselection()
        if not selection:
            messagebox.showwarning("Selection Error", "Please select a model first.")
            return

        model_line = self.model_listbox.get(selection[0])
        if "--> CURRENT" in model_line or "Created:" in model_line or model_line.strip() == "":
            messagebox.showwarning("Selection Error", "Please select a model file name.")
            return

        model_file = model_line.split(" <--")[0]

        # Check if it's the current model
        current_model = None
        if os.path.exists('current_model.txt'):
            with open('current_model.txt', 'r') as f:
                current_model = f.read().strip()

        if model_file == current_model:
            messagebox.showerror("Delete Error", "Cannot delete the currently loaded model.")
            return

        if not messagebox.askyesno("Confirm Delete", f"Are you sure you want to delete {model_file}?"):
            return

        try:
            os.remove(model_file)
            metadata_file = model_file.replace('.pkl', '_metadata.pkl')
            if os.path.exists(metadata_file):
                os.remove(metadata_file)

            self.update_status(f"Model {model_file} deleted successfully!")
            logger.log_model_action("MODEL_DELETED", model_file)
            self.refresh_model_list()
        except Exception as e:
            messagebox.showerror("Delete Error", f"Error deleting model: {e}")
            logger.log_error("MODEL_DELETE_ERROR", str(e))

    def create_model_from_original(self):
        """Create new model from original dataset."""
        if not messagebox.askyesno("Confirm", "This will run the training script on SMSSpamCollection.txt\n\nThis may take several minutes. Continue?"):
            return

        self.update_status("Creating new model from original dataset...")
        threading.Thread(target=self._create_model_from_original_thread).start()

    def _create_model_from_original_thread(self):
        """Thread for creating model from original dataset."""
        try:
            # Run preprocessing and training
            os.system('python preprocess.py')
            os.system('python train_model.py')
            self.load_current_model()
            self.root.after(0, lambda: self.update_status("New model created successfully!"))
            self.root.after(0, self.refresh_model_list)
            logger.log_model_action("MODEL_CREATED", "New model from original dataset")
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Creation Error", f"Error creating model: {e}"))
            self.root.after(0, lambda: self.update_status("Model creation failed"))
            logger.log_error("MODEL_CREATION_ERROR", str(e))

    def create_model_from_current(self):
        """Create new model from current dataset."""
        if not os.path.exists('user_feedback.log'):
            messagebox.showerror("Error", "No feedback data available. Use 'Update Dataset' first to incorporate feedback.")
            return

        if not messagebox.askyesno("Confirm", "This will create a new model from current dataset (including feedback)\n\nThis may take several minutes. Continue?"):
            return

        self.update_status("Creating new model from current dataset...")
        threading.Thread(target=self._create_model_from_current_thread).start()

    def _create_model_from_current_thread(self):
        """Thread for creating model from current dataset."""
        try:
            # Update dataset with feedback and retrain
            self.feedback_system.update_training_dataset()
            self.feedback_system.retrain_model()
            self.load_current_model()
            self.root.after(0, lambda: self.update_status("New model created with feedback data!"))
            self.root.after(0, self.refresh_model_list)
            logger.log_model_action("MODEL_CREATED", "New model from current dataset with feedback")
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Creation Error", f"Error creating model: {e}"))
            self.root.after(0, lambda: self.update_status("Model creation failed"))
            logger.log_error("MODEL_CREATION_ERROR", str(e))

    # Dataset update methods
    def update_dataset(self):
        """Update training dataset with user feedback."""
        if not os.path.exists('user_feedback.log'):
            messagebox.showerror("Error", "No feedback data available.")
            return

        self.update_status("Updating dataset with feedback...")
        self.update_btn.config(state='disabled')

        def update_thread():
            try:
                updated_file = self.feedback_system.update_training_dataset()
                self.root.after(0, lambda: self._show_dataset_update_result(updated_file))
                logger.log_model_action("DATASET_UPDATED", f"Feedback incorporated into {updated_file}")
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Update Error", f"Error updating dataset: {e}"))
                self.root.after(0, lambda: self.update_status("Dataset update failed"))
                logger.log_error("DATASET_UPDATE_ERROR", str(e))
            finally:
                self.root.after(0, lambda: self.update_btn.config(state='normal'))

        threading.Thread(target=update_thread).start()

    def _show_dataset_update_result(self, updated_file):
        """Show dataset update results."""
        self.dataset_status.config(state='normal')
        self.dataset_status.delete('1.0', tk.END)
        self.dataset_status.insert('1.0', f"Dataset updated successfully: {updated_file}\n\n")
        self.dataset_status.insert(tk.END, "The updated dataset is ready for retraining.")
        self.dataset_status.config(state='disabled')
        self.update_status("Dataset updated successfully")

    # Retrain methods
    def retrain_model(self):
        """Retrain the model."""
        if not os.path.exists('SMSSpamCollection_updated.txt'):
            messagebox.showerror("Error", "No updated dataset available. Use 'Update Dataset' first.")
            return

        if not messagebox.askyesno("Confirm", "This will retrain the model using the updated dataset\n\nThis may take several minutes. Continue?"):
            return

        self.update_status("Retraining model...")
        self.retrain_btn.config(state='disabled')
        self.retrain_progress.start()

        def retrain_thread():
            try:
                model_file = self.feedback_system.retrain_model()
                self.root.after(0, lambda: self._show_retrain_result(model_file))
                logger.log_model_action("MODEL_RETRAINED", model_file)
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Retrain Error", f"Error retraining model: {e}"))
                self.root.after(0, lambda: self.update_status("Model retraining failed"))
                logger.log_error("MODEL_RETRAIN_ERROR", str(e))
            finally:
                self.root.after(0, lambda: self.retrain_btn.config(state='normal'))
                self.root.after(0, lambda: self.retrain_progress.stop())

        threading.Thread(target=retrain_thread).start()

    def _show_retrain_result(self, model_file):
        """Show retrain results."""
        self.retrain_status.config(state='normal')
        self.retrain_status.delete('1.0', tk.END)
        self.retrain_status.insert('1.0', f"Model retrained successfully: {model_file}\n\n")
        self.retrain_status.insert(tk.END, "The new model has been loaded and is ready for use.")
        self.retrain_status.config(state='disabled')
        self.update_status("Model retrained successfully")
        self.load_current_model()

    # Stats methods
    def update_stats_display(self, event=None):
        """Update the statistics display."""
        log_type = self.log_type_var.get()
        self.stats_text.delete('1.0', tk.END)

        if log_type == 'echo_interactions':
            # Handle echo interactions log separately
            echo_log_path = os.path.join('.', 'echo_interactions.log')
            if os.path.exists(echo_log_path):
                try:
                    with open(echo_log_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                    recent_logs = lines[-5:] if len(lines) >= 5 else lines
                    for log_entry in recent_logs:
                        self.stats_text.insert(tk.END, log_entry)
                except Exception as e:
                    self.stats_text.insert(tk.END, f"Error reading echo log: {e}")
            else:
                self.stats_text.insert(tk.END, "No echo interactions log found.")
        else:
            recent_logs = logger.get_recent_logs(log_type, 5)

            if not recent_logs or recent_logs[0].startswith("Log file does not exist"):
                self.stats_text.insert(tk.END, "No entries found.")
            else:
                for log_entry in recent_logs:
                    self.stats_text.insert(tk.END, log_entry + '\n')

    # Benchmark methods
    def run_benchmark(self):
        """Run performance benchmark."""
        if not self.classifier.model:
            messagebox.showerror("Error", "No model loaded for benchmarking.")
            return

        self.update_status("Running performance benchmark...")
        self.benchmark_btn.config(state='disabled')
        self.benchmark_progress.start()

        def benchmark_thread():
            try:
                results = self._run_benchmark_internal()
                self.root.after(0, lambda: self._show_benchmark_results(results))
                logger.log_model_action("BENCHMARK_COMPLETED",
                                       f"Average time: {results['avg_time']:.3f}s, Avg confidence: {results['avg_confidence']:.3f}")
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Benchmark Error", f"Error running benchmark: {e}"))
                self.root.after(0, lambda: self.update_status("Benchmark failed"))
            finally:
                self.root.after(0, lambda: self.benchmark_btn.config(state='normal'))
                self.root.after(0, lambda: self.benchmark_progress.stop())

        threading.Thread(target=benchmark_thread).start()

    def _run_benchmark_internal(self):
        """Run the actual benchmark."""
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

        if results:
            avg_time = sum(r['time'] for r in results) / len(results)
            avg_confidence = sum(r['confidence'] for r in results) / len(results)
            spam_count = sum(1 for r in results if r['classification'] == 'spam')

            # Memory usage estimate
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024

            return {
                'avg_time': avg_time,
                'avg_confidence': avg_confidence,
                'total_time': total_time,
                'spam_count': spam_count,
                'total_messages': len(results),
                'memory_mb': memory_mb,
                'results': results
            }
        else:
            raise Exception("Benchmark failed - no valid results")

    def _show_benchmark_results(self, results):
        """Show benchmark results."""
        self.benchmark_results.config(state='normal')
        self.benchmark_results.delete('1.0', tk.END)

        result_text = "Benchmark Results:\n"
        result_text += "=" * 50 + "\n"
        result_text += f"Average classification time: {results['avg_time']:.3f} seconds\n"
        result_text += f"Average confidence: {results['avg_confidence']:.3f}\n"
        result_text += f"Total benchmark time: {results['total_time']:.3f} seconds\n"
        result_text += f"Spam detections: {results['spam_count']}/{results['total_messages']}\n"
        result_text += f"Memory usage: {results['memory_mb']:.1f} MB\n\n"

        result_text += "Individual Results:\n"
        result_text += "-" * 30 + "\n"
        for r in results['results']:
            result_text += f"Message: {r['message'][:40]}...\n"
            result_text += f"  Classification: {r['classification'].upper()}\n"
            result_text += f"  Confidence: {r['confidence']:.3f}\n"
            result_text += f"  Time: {r['time']:.3f}s\n\n"

        self.benchmark_results.insert('1.0', result_text)
        self.benchmark_results.config(state='disabled')
        self.update_status("Benchmark completed")

    def on_closing(self):
        """Handle application closing."""
        if self.current_model:
            logger.log_model_action("SYSTEM_SHUTDOWN", f"Current model: {self.current_model}")
        else:
            logger.log_model_action("SYSTEM_SHUTDOWN", "No model loaded")

        # Auto-disable echo logging on quit
        if echo_logger.echo_enabled:
            echo_logger.end_session()

        # Cleanup old logs
        logger.cleanup_old_logs()

        self.root.destroy()

def main():
    root = tk.Tk()
    app = SMSSpamDetectorGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()