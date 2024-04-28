# import data_processing
# from sklearn.feature_extraction.text import TfidfVectorizer


# # def test_model(model, test_data):
# #     # Preprocess data
# #     preprocessed_data = data_processing.preprocess_data(test_data)
# #     # Vectorize text data
# #     X = [entry['text'] for entry in preprocessed_data]
# #     vectorizer = TfidfVectorizer()
# #     X_vectorized = vectorizer.transform(X)
# #     # Make predictions
# #     predictions = model.predict(X_vectorized)
# #     # Output predictions or save them to a file
# #     for entry, prediction in zip(preprocessed_data, predictions):
# #         print("Sentence:", entry['text'])
# #         print("Target:", entry['target'])
# #         print("Predicted Category:", prediction)

# # def test_model(model, vectorizer, test_data):
# #     # Preprocess data
# #     preprocessed_data = data_processing.preprocess_data(test_data)
# #     # Vectorize text data
# #     X = [entry['text'] for entry in preprocessed_data]
# #     X_vectorized = vectorizer.transform(X)
# #     # Make predictions
# #     predictions = model.predict(X_vectorized)
# #     # Output predictions or save them to a file
# #     for entry, prediction in zip(preprocessed_data, predictions):
# #         print("Sentence:", entry['text'])
# #         print("Target:", entry['target'])
# #         print("Predicted Category:", prediction)
# # def predict_opinions(model, vectorizer, data):
# #     """
# #     Predicts opinions for a given dataset using a trained model and vectorizer.
# #     """
# #     X_data = [entry['text'] for entry in data]
# #     X_data_preprocessed = data_processing.preprocess_text(X_data)
# #     X_vectorized = vectorizer.transform(X_data_preprocessed)
# #     predictions = model.predict(X_vectorized)
# #     return predictions
# import numpy as np

# # def predict_opinions(model, vectorizer, test_data, mlb):
# #     X_data = []
# #     for entry in test_data:
# #         sentences = entry.get('sentences', [])
# #         for sentence in sentences:
# #             text = sentence.get('text')
# #             if text:
# #                 X_data.append(text)
# #             else:
# #                 print("Error: 'text' key not found in the entry.")

# #     if not X_data:
# #         print("Error: No valid texts found in the test data.")
# #         return None

# #     X_data_vectorized = vectorizer.transform(X_data)
# #     y_pred = model.predict(X_data_vectorized)

# #     predictions = []
# #     index = 0
# #     for entry in test_data:
# #         entry_predictions = []
# #         for sentence in entry.get('sentences', []):
# #             if index < len(y_pred):
# #                 predicted_opinions = mlb.inverse_transform([y_pred[index]])
# #                 opinions = []
# #                 for opinion_label in predicted_opinions[0]:
# #                     label_parts = opinion_label.split('#')
# #                     if len(label_parts) == 5:
# #                         target, category, polarity, from_, to = label_parts
# #                         opinion = {
# #                             'target': target,
# #                             'category': category,
# #                             'polarity': polarity,
# #                             'from': from_,
# #                             'to': to
# #                         }
# #                         opinions.append(opinion)
# #                     else:
# #                         print(f"Warning: Invalid label format for '{opinion_label}'. Skipping.")
# #                 entry_predictions.append(opinions)
# #             else:
# #                 print("Error: Index out of range for y_pred.")
# #             index += 1
# #         predictions.append(entry_predictions)

# #     return predictions

# def predict_opinions(model, vectorizer, test_data, mlb):
#     predicted_opinions_list = []

#     X_data = []
#     for entry in test_data:
#         sentences = entry.get('sentences', [])
#         for sentence in sentences:
#             text = sentence.get('text')
#             if text:
#                 X_data.append(text)

#     if not X_data:
#         print("Error: No valid texts found in the test data.")
#         return None

#     X_data_vectorized = vectorizer.transform(X_data)
#     y_pred = model.predict(X_data_vectorized)


#     index = 0
#     for entry in test_data:
#         predicted_opinions_for_review = []
#         for sentence in entry.get('sentences', []):
#             if index < len(y_pred):
#                 predicted_opinions = []
                # y_pred_array = np.array([y_pred[index]])  # Convert to NumPy array
                # for opinion_label in mlb.inverse_transform(y_pred_array)[0]:
                    
                #     parts = opinion_label.split('#')
                 

                #     if len(parts) == 5:
                #         target, category, polarity, from_, to = parts
                #         predicted_opinions.append({
                #             'target': target,
                #             'category': category,
                #             'polarity': polarity,
                #             'from': from_,
                #             'to': to
                #         })
                #         if not predicted_opinions:
                #             # If no opinion is predicted, provide a default value or indication
                #             predicted_opinions = [{'target': 'Null', 'category': 'None', 'polarity': 'None', 'from': 0, 'to': 0}]
                #     else:
                #         print(f"Error: Invalid opinion label format - {opinion_label}")
#                 # for opinion_label in mlb.inverse_transform(y_pred_array)[0]:
#                 #     target, category, polarity, from_, to = opinion_label.split('#')
#                 #     predicted_opinions.append({
#                 #         'target': target,
#                 #         'category': category,
#                 #         'polarity': polarity,
#                 #         'from': from_,
#                 #         'to': to
#                 #     })
#                 # predicted_opinions_for_review.append(predicted_opinions)
#             else:
#                 print("Error: Index out of range for y_pred.")
#             index += 1
#         predicted_opinions_list.append(predicted_opinions_for_review)

#         inconsistent_labels = check_opinion_labels_format(predicted_opinions_list)
#         if inconsistent_labels:
#             print("Inconsistent opinion labels:")
#             for label in inconsistent_labels:
#                 print(label)
#         else:
#             print("No inconsistent opinion labels found.")

#     return predicted_opinions_list

# import re


# def check_opinion_label_format(opinion_label):
#     """
#     Check if the opinion label follows the expected format.
#     Expected format: ASPECT#CATEGORY
#     """
#     pattern = r'^\w+#\w+$'  # Regular expression pattern for ASPECT#CATEGORY format
#     return re.match(pattern, opinion_label) is not None

# def check_opinion_labels_format(predictions):
#     inconsistent_labels = []
#     for review in predictions:
#         for sentence in review:
#             for opinion in sentence:
#                 if not all(key in opinion for key in ['category', 'polarity']):
#                     inconsistent_labels.append(opinion)
#     return inconsistent_labels



# import numpy as np

# def predict_opinions(model, vectorizer, test_data, mlb):
#     predicted_opinions_list = []

#     X_data = []
#     for entry in test_data:
#         sentences = entry.get('sentences', [])
#         for sentence in sentences:
#             text = sentence.get('text')
#             if text:
#                 X_data.append(text)

#     if not X_data:
#         print("Error: No valid texts found in the test data.")
#         return None

#     X_data_vectorized = vectorizer.transform(X_data)
#     y_pred = model.predict(X_data_vectorized)

#     index = 0
#     for entry in test_data:
#         predicted_opinions_for_review = []
#         for sentence in entry.get('sentences', []):
#             predicted_opinions = []
#             if index < len(y_pred):
#                 y_pred_slice = np.array([y_pred[index]])  # Convert to NumPy array
#                 for opinion_label in mlb.inverse_transform(y_pred_slice)[0]:
#                     # Split the opinion label by '#'
#                     opinion_parts = opinion_label
#                     print("opinion_parts", opinion_parts)
#                     if len(opinion_parts) == 5:  # Check if the split produced the expected number of parts
#                         target, category, polarity, from_, to = opinion_parts
#                     else:
#                         # Handle the case where the split did not produce the expected number of parts
#                         print("Warning: Invalid opinion label format -", opinion_label)
#                         # Assign default values for 'from_' and 'to'
#                         # category, polarity = opinion_parts[:2]
#                         # target = 'Null'
#                         # # target, category, polarity = opinion_parts[:3]
#                         # from_, to = '0', '0'
#                         category = opinion_parts[1]
#                         # target = opinion_parts[0]
#                         # target = 'Null'


#                         if len(opinion_parts) > 2:
#                             polarity = opinion_parts[2]
#                         else:
#                             polarity = 'neutral'  # Assign a default polarity if not provided

#                         if len(opinion_parts) > 3 and opinion_parts[0] == 'NULL':
#                             from_, to = '0', '0'  # Assign default values for 'from_' and 'to'
#                         else:
#                             from_, to = opinion_parts[3] , opinion_parts[4]

#                         # from_, to = '0', '0'  # Assign default values for 'from_' and 'to'
                   


#                     predicted_opinions.append({
#                         'target': target,
#                         'category': category,
#                         'polarity': polarity,
#                         'from': from_,
#                         'to': to
#                     })
#             else:
#                 print("Error: Index out of range for y_pred.")
#             index += 1
#             predicted_opinions_for_review.append(predicted_opinions)
#         predicted_opinions_list.append(predicted_opinions_for_review)

#     return predicted_opinions_list


import numpy as np

# def predict_opinions(model, vectorizer, test_data, mlb):
#     predicted_opinions_list = []

#     X_data = []
#     for entry in test_data:
#         sentences = entry.get('sentences', [])
#         for sentence in sentences:
#             text = sentence.get('text')
#             if text:
#                 X_data.append(text)

#     if not X_data:
#         print("Error: No valid texts found in the test data.")
#         return None

#     X_data_vectorized = vectorizer.transform(X_data)
#     y_pred = model.predict(X_data_vectorized)

#     index = 0
#     for entry in test_data:
#         predicted_opinions_for_review = []
#         for sentence in entry.get('sentences', []):
#             predicted_opinions = []
#             if index < len(y_pred):
#                 y_pred_slice = np.array([y_pred[index]])
#                 for opinion_label in mlb.inverse_transform(y_pred_slice)[0]:
#                     opinion_parts = opinion_label
#                     print("opinion_parts", opinion_parts)
#                     if len(opinion_parts) == 5:
#                         target, category, polarity, from_, to = opinion_parts
#                     else:
#                         print("Warning: Invalid opinion label format -", opinion_label)
#                         category = opinion_parts[0]
#                         target = 'NULL'
#                         if len(opinion_parts) > 1:
#                             polarity = opinion_parts[1]
#                         else:
#                             polarity = 'neutral'
#                         from_, to = '0', '0'

#                     predicted_opinions.append({
#                         'target': target,
#                         'category': category,
#                         'polarity': polarity,
#                         'from': from_,
#                         'to': to
#                     })
#             else:
#                 print("Error: Index out of range for y_pred.")
#             index += 1
#             predicted_opinions_for_review.append(predicted_opinions)
#         predicted_opinions_list.append(predicted_opinions_for_review)

#     return predicted_opinions_list


# Function to make predictions for each aspect of the opinion
# model_testing.py

# def predict_opinions(clf_category, clf_polarity, clf_target, clf_from, clf_to, vectorizer, test_data):
#     predictions = []
    
#     # Initialize VADER sentiment analyzer
#     # nltk.download('vader_lexicon')  # Download VADER lexicon
    # sid = SentimentIntensityAnalyzer()
    
#     for entry in test_data:
#         sentences = entry.get('sentences', [])
#         for sentence in sentences:
#             text = sentence.get('text')
#             if text:
#                 # Sentiment analysis using VADER
#                 sentiment_scores = sid.polarity_scores(text)
#                 # Determine sentiment polarity
#                 sentiment_polarity = 'neutral'
#                 if sentiment_scores['compound'] > 0.05:
#                     sentiment_polarity = 'positive'
#                 elif sentiment_scores['compound'] < -0.05:
#                     sentiment_polarity = 'negative'

                
#                 # Make predictions for other aspects
#                 X_test = vectorizer.transform([text])
#                 # print("X_test", X_test)
#                 # print('polairty', sentiment_polarity)
#                 # print('category', clf_category.predict(X_test))
#                 # print('target', clf_target.predict(X_test))
#                 # print('from', clf_from.predict(X_test))
#                 # print('to', clf_to.predict(X_test))
#                 category_pred = clf_category.predict(X_test)[0]
#                 target_pred = clf_target.predict(X_test)[0]
#                 from_pred = clf_from.predict(X_test)[0]
#                 to_pred = clf_to.predict(X_test)[0]
#                 polarity = clf_polarity.predict(X_test)[0]
                
#                 opinion = {
#                     'target': target_pred,
#                     'category': category_pred,
#                     'polarity': polarity,
#                     # 'polarity': sentiment_polarity,
#                     'from': from_pred,
#                     'to': to_pred
#                 }
#                 predictions.append(opinion)
#     return predictions

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import numpy as np



def predict_opinions(category_model, polarity_model, target_model, from_model, to_model, vectorizer, test_data):
        # Initialize VADER sentiment analyzer
    nltk.download('vader_lexicon') 
    predictions = []
    sid = SentimentIntensityAnalyzer()
    for entry in test_data:
        sentences = entry.get('sentences', [])
        for sentence in sentences:
            text = sentence.get('text')
            if text:
                # Sentiment analysis using VADER
                sentiment_scores = sid.polarity_scores(text)

                # Determine sentiment polarity
                sentiment_polarity = 'neutral'
                if sentiment_scores['compound'] > 0.05:
                    sentiment_polarity = 'positive'
                elif sentiment_scores['compound'] < -0.05:
                    sentiment_polarity = 'negative'

                # print("sentiment_polarity", sentiment_polarity)
                # print("sentiment scores", sentiment_scores)
                # Make predictions for each aspect
                category_pred = category_model.predict(vectorizer.transform([text]))
                polarity_pred = polarity_model.predict(vectorizer.transform([text]))
                target_pred = target_model.predict(vectorizer.transform([text]))

                # # Calculate the positions of the target word within the sentence
                # try:
                #     from_pred = text.lower().index(target_pred[0].lower())
                #     to_pred = from_pred + len(target_pred[0]) - 1
                # except ValueError:
                #     # Handle case where target word is not found
                #     target_pred[0] = 'NULL'
                #     from_pred = 0
                #     to_pred = 0
                  # Calculate the positions of the target word within the sentence
                target_lower = target_pred[0].lower()
                text_lower = text.lower()
                from_pred = text_lower.find(target_lower)
                to_pred = from_pred + len(target_lower) - 1

                # Handle case where target word is not found
                if from_pred == -1:
                    target_pred[0] = 'NULL'
                    from_pred = 0
                    to_pred = 0


                # # Merge predictions into one opinion label
                opinion = {
                    'target': target_pred[0],
                    'category': category_pred[0],
                    'polarity': sentiment_polarity,
                    # 'polarity': polarity_pred[0],
                    'from': from_pred,
                    'to': to_pred
                }
                sentenceId =  sentence['id'],
                # print("text", text)
                # print("target_pred", target_pred[0])
                # print("from_pred", from_pred)
                # print("to_pred", to_pred)
                # print("opinion", opinion    )
                # total_opinions = []
                # predictions.append(opinion)
                # print("sentenceId", sentenceId)
                predictions.append({'sentence': sentenceId, 'opinion': opinion})
    return predictions




