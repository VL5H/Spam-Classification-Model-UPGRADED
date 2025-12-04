import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import string

# Download NLTK data if not present
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

def load_dataset(file_path='SMSSpamCollection.txt'):
    """Load the SMS dataset from tab-separated file."""
    df = pd.read_csv(file_path, sep='\t', header=None, names=['label', 'message'])
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    return df

def clean_text(text):
    """Clean and preprocess text data."""
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    # Join back to string
    return ' '.join(tokens)

def preprocess_data(df):
    """Apply text cleaning to the dataset."""
    df['clean_message'] = df['message'].apply(clean_text)
    return df

def balance_dataset(df):
    """Handle class imbalance by oversampling minority class."""
    # Separate majority and minority classes
    df_majority = df[df.label == 0]
    df_minority = df[df.label == 1]

    # Oversample minority class
    df_minority_upsampled = resample(df_minority,
                                     replace=True,
                                     n_samples=len(df_majority),
                                     random_state=42)

    # Combine majority class with upsampled minority class
    df_balanced = pd.concat([df_majority, df_minority_upsampled])

    return df_balanced

def vectorize_text(df, max_features=5000):
    """Convert text to TF-IDF vectors."""
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
    X = vectorizer.fit_transform(df['clean_message'])
    y = df['label'].values

    return X, y, vectorizer

def split_data(X, y, test_size=0.2, val_size=0.2):
    """Split data into train, validation, and test sets."""
    # First split: train and temp (val+test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=test_size + val_size, random_state=42, stratify=y
    )

    # Second split: val and test from temp
    val_ratio = val_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=1-val_ratio, random_state=42, stratify=y_temp
    )

    return X_train, X_val, X_test, y_train, y_val, y_test

def save_preprocessed_data(X_train, X_val, X_test, y_train, y_val, y_test, vectorizer):
    """Save preprocessed data and vectorizer for later use."""
    import joblib

    # Save data as sparse matrices
    joblib.dump(X_train, 'X_train.pkl')
    joblib.dump(X_val, 'X_val.pkl')
    joblib.dump(X_test, 'X_test.pkl')
    joblib.dump(y_train, 'y_train.pkl')
    joblib.dump(y_val, 'y_val.pkl')
    joblib.dump(y_test, 'y_test.pkl')
    joblib.dump(vectorizer, 'vectorizer.pkl')

    print("Preprocessed data saved successfully.")

if __name__ == "__main__":
    # Load and preprocess dataset
    print("Loading dataset...")
    df = load_dataset()

    print("Preprocessing text...")
    df = preprocess_data(df)

    print("Balancing dataset...")
    df_balanced = balance_dataset(df)

    print("Vectorizing text...")
    X, y, vectorizer = vectorize_text(df_balanced)

    print("Splitting data...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    print("Saving preprocessed data...")
    save_preprocessed_data(X_train, X_val, X_test, y_train, y_val, y_test, vectorizer)

    print(f"Dataset shape: {df.shape}")
    print(f"Balanced dataset shape: {df_balanced.shape}")
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print("Preprocessing complete!")