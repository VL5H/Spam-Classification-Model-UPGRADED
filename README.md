# Spam-Classification-Model-UPGRADED
A heavily upgraded and redesigned version of my previous spam message classification model --> [Spam-Message-Classification-Model](https://github.com/VL5H/Spam-Message-Classification-Model)

This model also classifies SMS messages as either "spam" or "ham" with the following critical improvements:
1. The original dataset of 5000+ samples was heavily skewed towards "ham" with about 75% of the messages being examples of normal messages. This imbalance was equalized and the total number of samples was boosted to around 21,000 examples of both "spam" and "ham" messages.
2. The previously used Naive Bayes model was deemed too simplistic and was replaced with a machine learning based approach. 
3. Both a CLI and a GUI was added to improve user experience; users can now enter their own messages to be classified as well as control other functions. 
4. Logging system was also improved now logging model performance, errors, actions, and confidence per classification.
5. Users can now give feedback to the model and retrain the model as needed.

The new classification model is a combination of 3 classic ML algorithms implemented via the scikit-learn library: Logistic Regression, Support Vector Machine (SVM), and a Random Forest Classifier. All 3 models are trained on the dataset and the best performer's response (based on F1 score) is selected as the final output. These 3 models were selected for their greater performance with limited resources and smaller file sizes.

More details about the above 3 models can be found here:
1. Logistic Regression Info - [sk-learn Logistic Regression Models](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
2. SVM - [sk-learn General SVMs](https://scikit-learn.org/stable/modules/svm.html) & [sk-learn SVCs](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)
3. Random Forest Classifier - [sk-learn Random Forest Classifiers](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)


How the model works:
1. Preprocessing - 
    - Dataset is loaded from "SMSSpamCollection.txt" or ""SMSSpamCollection_updated.txt" as per user selection.
    - Samples are cleaned (symbols and numbers removed, all lowercased) and tokenized via NLTK. Stopwords are also removed before the sample is re-combined as a single string.
    - Data is checked for class imbalances, the samples from the minority class are cloned if necessary to equalize the ration between spam and ham.
    - Feature extraction is performed: text is vectorized via a TF-IDF ([Term Frequency-Inverse Document Frequency](https://www.geeksforgeeks.org/machine-learning/understanding-tf-idf-term-frequency-inverse-document-frequency/)) algorithm. Vocabulary is limited to the top 5,000 words including both single and 2-word phrases.
    - Data set is split into: 60% for training, 20% for validation and model output selection, 20% for testing. Class balance is the same throughout all sub-sets.
2. Model Training -
    - As stated above 3 models are trained on the training set: logistic regression model, SVM, and random forest.
    - Each model is then evaluated for accuracy, precision, recall, F1, and per class performance.
    - The model with the highest F1 score is then selected as the final model (usually SVM) and saved along with its vectorizer as a ".pkl" file.
3. Classification -
    - Messages inputted to the model undergo the same preprocessing process as above.
    - Probability score for both classes is calculated along with the model's confidence score (sum of both probabilities)
    - The higher probability score becomes the final output alongside the model's confidence score.
    - If confidence score is low (< 80%), user has the option of providing feedback by manually classifying the message. This is then added to the model's training data.
4. Logging -
    - The model keeps the following running ".log" files (any log entries that are older than 30-days old will be automatically cleaned up at runtime):
       - "model_actions.log" which is a record of any and all actions performed on or by the model including training, message classification, etc.
       - "user_feedback.log" which is a record of any and all user feedback given to the model.
       - "error_reports.log" which is a record of any errors that the model system may encounter including model-not-found errors, model corrupted errors, etc.
       - "echo_interactions.log" which is an optionally activated CLI debugging tool that mirrors any and all text that shows up on the terminal.
5. Other Features -
    - The user has the option of running benchmarks on the current model at any time via the GUI/CLI or full system diagnostics via running the "test_system.py" file.
    - The user has the ability to create, update, retrain, select, and delete models at any time via the GUI/CLI.
    - The user has the ability to view the last 5 log entries from all logs at any time via the GUI/CLI.
  
Installation/Set-Up:
1. Download the following files:
    - sms_spam_detector.py - Main CLI interface
    - inference_engine.py - Classification engine
    - preprocess.py - Data preprocessing pipeline
    - train_model.py - Model training script
    - feedback_system.py - User feedback collection system
    - logging_system.py - Logging infrastructure
    - model_management.py - Model lifecycle management
    - benchmark.py - Performance testing
    - test_system.py - Full system testing
    - sms_spam_detector_gui.py - Main GUI interface
    - requirements.txt - List of python libraries/packages to install
    - SMSSpamCollection.txt - Updated/enhanced dataset
2. Ensure that all files are in the same directory.

Running/Usage:
1. Create/activate your virtual environment (you will need Python 3.8 or higher) and run the command: ```pip install -r requirements.txt```
2. Select "preprocess.py" and run it to process the data set and generate ".pkl" files for training.
3. Select "train_model.py" to train and generate the initial model file.
4. Select either "sms_spam_detector.py" for CLI or "sms_spam_detector_gui.py" for GUI and run it to start-up the classification and logging systems.
5. The following files should get auto-created on first run -
    - X_train.pkl - Training feature matrix
    - X_val.pkl - Validation feature matrix
    - X_test.pkl - Testing feature matrix
    - y_train.pkl - Training labels
    - y_val.pkl - Validation labels
    - y_test.pkl - Testing labels
    - vectorizer.pkl - Text vectorization model
    - "model_name".pkl - Actual trained model
    - "model_name_metadata".pkl - Model metadata file
    - current_model.txt - Reference to the current active model
    - model_actions.log - Model training, classification, etc. log
    - user_feedback.log - User feedback log
    - error_reports.log - System errors log
    - echo_interactions.log - Optional terminal activity log
    - echo_session.txt - Current echo logging session number
    - echo_state.txt - Echo logging enabled/disabled state
    - SMSSpamCollection_updated.txt - Updated dataset with user feedback added to it (will only generate if user-feedback is created first)
6. The model should average somewhere around 92% accuracy with its classifications and will constantly learn via user feedback.
