import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

def load_data(file_path):
    column_names = [
        "Class", "Attribute1", "Attribute2", "Attribute3", "Attribute4", "Attribute5"
    ]
    df = pd.read_csv(file_path, names=column_names, na_values='?')
    return df

def preprocess_data(df):
    X = df.drop('Class', axis=1)
    y = df['Class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    model = HistGradientBoostingClassifier()
    model.fit(X_train, y_train)
    joblib.dump(model, 'thyroid_model.pkl')
    return model

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    # Print classification report
    print("Classification Report:\n")
    print(classification_report(y_test, predictions))
    
    # Print confusion matrix
    cm = confusion_matrix(y_test, predictions)
    print("\nConfusion Matrix:\n")
    print(cm)


if __name__ == "__main__":
    df = load_data('https://archive.ics.uci.edu/ml/machine-learning-databases/thyroid-disease/new-thyroid.data')
    X_train, X_test, y_train, y_test = preprocess_data(df)
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)