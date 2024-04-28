from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer

# import data_processing

# def evaluate_model(model, test_data):
#     # Preprocess data
#     preprocessed_data = data_processing.preprocess_data(test_data)
#     # Split data into features and labels
#     X = [entry['text'] for entry in preprocessed_data]
#     y_true = [entry['category'] for entry in preprocessed_data]
#     # Vectorize text 
#     vectorizer = TfidfVectorizer()
#     X_vectorized = vectorizer.transform(X)
#     # Make predictions
#     y_pred = model.predict(X_vectorized)
#     # Evaluate model performance
#     accuracy = accuracy_score(y_true, y_pred)
#     precision = precision_score(y_true, y_pred, average='weighted')
#     recall = recall_score(y_true, y_pred, average='weighted')
#     f1 = f1_score(y_true, y_pred, average='weighted')
#     # Print evaluation metrics
#     print("Accuracy:", accuracy)
#     print("Precision:", precision)
#     print("Recall:", recall)
#     print("F1 Score:", f1)
import data_processing

# def evaluate_model(model, vectorizer, test_data):
#     print("Test data length:", len(test_data))
#     # Preprocess data
#     preprocessed_data = data_processing.preprocess_data(test_data)
#     # Split data into features and labels
#     X = [entry['text'] for entry in preprocessed_data]
#     y_true = [entry['category'] for entry in preprocessed_data]
#     print("Test data length after preprocessing:", len(X))
#     # Vectorize text data
#     X_vectorized = vectorizer.transform(X)
#     # Make predictions
#     y_pred = model.predict(X_vectorized)
#     # Evaluate model performance
#     accuracy = accuracy_score(y_true, y_pred)
#     precision = precision_score(y_true, y_pred, average='weighted')
#     recall = recall_score(y_true, y_pred, average='weighted')
#     f1 = f1_score(y_true, y_pred, average='weighted')
#     # Print evaluation metrics
#     print("Accuracy:", accuracy)
#     print("Precision:", precision)
#     print("Recall:", recall)
#     print("F1 Score:", f1)


# from sklearn.preprocessing import MultiLabelBinarizer

# def evaluate_model(model, vectorizer, test_data):
#     X_test = []
#     for entry in test_data:
#         # Adjust this part according to your test data structure
#         # Here, assuming each entry has a 'sentences' field containing a list of sentences,
#         # and each sentence has a 'text' field
#         sentences = entry.get('sentences', [])
#         for sentence in sentences:
#             text = sentence.get('text')
#             if text:
#                 X_test.append(text)
    
#     # Vectorize test data
#     X_test_vectorized = vectorizer.transform(X_test)
    
#     # Predict labels
#     y_pred = model.predict(X_test_vectorized)
    
#     # Assuming mlb is the MultiLabelBinarizer used during training
#     # In case you used a different vectorizer for test data, adjust accordingly
#     mlb = MultiLabelBinarizer()
#     y_pred_labels = mlb.inverse_transform(y_pred)
    
#     # Print predictions
#     for i, entry in enumerate(test_data):
#         print(f"Predictions for entry {i}: {y_pred_labels[i]}")

import numpy as np

def evaluate_model(model, vectorizer, test_data, mlb):
    X_test = []
    for entry in test_data:
        sentences = entry.get('sentences', [])
        for sentence in sentences:
            text = sentence.get('text')
            if text:
                X_test.append(text)
                

    X_test_vectorized = vectorizer.transform(X_test)

    y_pred = model.predict(X_test_vectorized)
    y_pred = np.array(y_pred)

    try:
        for i, entry in enumerate(test_data):
            if i < len(y_pred):
                y_pred_slice = y_pred[i:i + 1]
                if len(y_pred_slice[0]) == 0:
                    print(f"No predictions available for entry {i}.")
                else:
                    y_pred_labels = mlb.inverse_transform(y_pred_slice)
                    print(f"Predictions for entry {i}: {y_pred_labels[0]}")
            else:
                print(f"No predictions available for entry {i}.")
    except Exception as e:
        print("Error occurred during prediction:", e)