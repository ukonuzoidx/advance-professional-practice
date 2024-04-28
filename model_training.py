# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.svm import SVC
# from sklearn.metrics import classification_report
# import data_processing

# def train_model(train_data):
#     # Preprocess data
#     preprocessed_data = data_processing.preprocess_data(train_data)
#     # Split data into features and labels
#     X = [entry['text'] for entry in preprocessed_data]
#     y = [entry['category'] for entry in preprocessed_data]
#     # Vectorize text data
#     vectorizer = TfidfVectorizer()
#     X_vectorized = vectorizer.fit_transform(X)
#     # Split data into train and test sets
#     X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)
#     # Train machine learning model
#     model = SVC()
#     model.fit(X_train, y_train)
#     # Evaluate model
#     y_pred = model.predict(X_test)
#     print(classification_report(y_test, y_pred))

#     # Return trained model
#     return model
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import data_processing

# def train_model(train_data, vectorizer):
#     # Preprocess data
#     preprocessed_data = data_processing.preprocess_data(train_data)
#     # Split data into features and labels
#     X = [entry['text'] for entry in preprocessed_data]
#     y = [entry['category'] for entry in preprocessed_data]
#     # Vectorize text data
#     X_vectorized = vectorizer.transform(X)
#     # Split data into train and test sets
#     X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)
#     # Train machine learning model
#     model = SVC()
#     model.fit(X_train, y_train)
#     # Evaluate model
#     y_pred = model.predict(X_test)
#     print(classification_report(y_test, y_pred))

#     # Return trained model
#     return model
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# def train_model(data, vectorizer):
#     """
#     Trains a machine learning model using the provided data and vectorizer.
#     Returns the trained model.
#     """
#     X_train = [entry['text'] for entry in data if entry['opinions'] is not None]
#     y_train = [entry['opinions'] for entry in data if entry['opinions'] is not None]
#     if y_train:
#         pipeline = Pipeline([('vectorizer', vectorizer), ('clf', LogisticRegression())])
#         pipeline.fit(X_train, y_train)
#         return pipeline
#     else:
#         print("No labeled data found. Cannot train the model.")
#         return None

# from sklearn.preprocessing import MultiLabelBinarizer
# from sklearn.multioutput import MultiOutputClassifier
# from sklearn.linear_model import LogisticRegression

# def train_model(data, vectorizer):
#     # Extract texts and opinions
#     X_train = []
#     y_train = []
#     for entry in data:
#         sentences = entry.get('sentences', [])
#         for sentence in sentences:
#             text = sentence.get('text')
#             opinions = sentence.get('opinions')
#             if text and opinions is not None:
#                 X_train.append(text)
#                 # Assuming you want to extract opinion labels
#                 # from the XML data structure
#                 # Modify this part according to your data structure
#                 # Here, I'm assuming that 'Opinion' elements have a 'category' attribute
#                 # and we extract that as the opinion label
#                 # in the opinions get category and polarity and append to y_train
#                 opinion_labels = [opinion.get('category') for opinion in opinions]
#                 y_train.append(opinion_labels)

#     # Vectorize texts
#     X_train_vectorized = vectorizer.transform(X_train)
    
    # # Use MultiLabelBinarizer to convert list of labels into binary representation
    # mlb = MultiLabelBinarizer()
    # y_train = mlb.fit_transform(y_train)

    # # Train model (dummy example)
    # # Replace this with your actual model training code
    # # Dummy example: using a multi-label classifier (e.g., MultiOutputClassifier with LogisticRegression)
    # model = MultiOutputClassifier(LogisticRegression())
    # model.fit(X_train_vectorized, y_train) 

    # return model, mlb  # Corrected variable name here to return 'mlb' instead of 'MLB'

# import tensorflow as tf
# from keras.layers import Dense, Input
# from keras.models import Model
# from sklearn.preprocessing import MultiLabelBinarizer
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.multioutput import MultiOutputClassifier

# def train_model(data, vectorizer):
#     X_train = []
#     y_train = []
#     for entry in data:
#         sentences = entry.get('sentences', [])
#         for sentence in sentences:
#             text = sentence.get('text')
#             opinions = sentence.get('opinions')
#             if text and opinions is not None:
#                 X_train.append(text)
#                 sentence_labels = []
#                 for opinion in opinions:
#                     target = opinion.get('target', 'NULL')
#                     category = opinion.get('category', 'NULL')
#                     polarity = opinion.get('polarity', 'neutral')
#                     from_ = opinion.get('from', '0')
#                     to = opinion.get('to', '0')
#                     # sentence_labels.append((target, category, polarity))
#                     sentence_labels.append((target, category, polarity, from_, to))
#                 y_train.append(sentence_labels)

#     X_train_vectorized = vectorizer.transform(X_train)
   
#     # Use MultiLabelBinarizer to convert list of labels into binary representation
#     mlb = MultiLabelBinarizer()
#     y_train = mlb.fit_transform(y_train)

#     # Train model (dummy example)
#     # Replace this with your actual model training code
#     # Dummy example: using a multi-label classifier (e.g., MultiOutputClassifier with LogisticRegression)
#     model = MultiOutputClassifier(LogisticRegression())
#     model.fit(X_train_vectorized, y_train) 

#     return model, mlb  # Corrected variable name here to return 'mlb' instead of 'MLB'
# from sklearn.linear_model import LogisticRegression
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.preprocessing import MultiLabelBinarizer
# from sklearn.multioutput import MultiOutputClassifier
# import numpy as np

# def train_models(data, vectorizer):
#     # Extract texts and opinions
#     X_train = []
#     categories_train = []
#     polarities_train = []
#     targets_train = []
#     from_positions_train = []
#     to_positions_train = []

#     for entry in data:
#         sentences = entry.get('sentences', [])
#         for sentence in sentences:
#             text = sentence.get('text')
#             opinions = sentence.get('opinions', [])
#             if text and opinions:
#                 for opinion in opinions:
#                     category = opinion.get('category')
#                     polarity = opinion.get('polarity')
#                     target = opinion.get('target')
#                     from_ = opinion.get('from')
#                     to = opinion.get('to')

#                     X_train.append(text)
#                     categories_train.append(category)
#                     polarities_train.append(polarity)
#                     targets_train.append(target)
#                     from_positions_train.append(from_)
#                     to_positions_train.append(to)

#     # Vectorize texts
#     X_train_vectorized = vectorizer.transform(X_train)

#     # Train models for each aspect
#     category_model = LogisticRegression()
#     category_model.fit(X_train_vectorized, categories_train)

#     polarity_model = LogisticRegression()
#     polarity_model.fit(X_train_vectorized, polarities_train)

#     target_model = LogisticRegression()
#     target_model.fit(X_train_vectorized, targets_train)

#     from_to_model = LogisticRegression()
#     from_to_model.fit(X_train_vectorized, np.array(list(zip(from_positions_train, to_positions_train))))

#     return category_model, polarity_model, target_model, from_to_model

from sklearn.linear_model import LogisticRegression

def train_model(data, vectorizer):
    texts = []
    categories = []
    polarities = []
    targets = []
    from_positions = []
    to_positions = []
    

    for entry in data:
        sentences = entry.get('sentences', [])
        for sentence in sentences:
            text = sentence.get('text')
            opinions = sentence.get('opinions', [])
            if text and opinions:
                for opinion in opinions:
                    category = opinion.get('category')
                    polarity = opinion.get('polarity')
                    target = opinion.get('target')
                    from_ = opinion.get('from')
                    to = opinion.get('to')

                    texts.append(text)
                    categories.append(category)
                    polarities.append(polarity)
                    targets.append(target)
                    from_positions.append(from_)
                    to_positions.append(to)

    X = vectorizer.fit_transform(texts)

    clf_category = LogisticRegression()
    clf_polarity = LogisticRegression()
    clf_target = LogisticRegression()
    clf_from = LogisticRegression()
    clf_to = LogisticRegression()

    clf_category.fit(X, categories)
    clf_polarity.fit(X, polarities)
    clf_target.fit(X, targets)
    clf_from.fit(X, from_positions)
    clf_to.fit(X, to_positions)

    return clf_category, clf_polarity, clf_target, clf_from, clf_to