# import data_processing
# import model_training
# import model_evaluation
# import model_testing
# from sklearn.feature_extraction.text import TfidfVectorizer

# if __name__ == "__main__":
#     # Parse training data
#     train_data = data_processing.parse_xml('datasets/train.xml')
#     # Train model
#     vectorizer = TfidfVectorizer()
#     trained_model = model_training.train_model(train_data)
#     # Parse test data
#     test_data = data_processing.parse_xml('datasets/test.xml')
#     # Evaluate model
#     model_evaluation.evaluate_model(trained_model, test_data)
#     # Test model
#     model_testing.test_model(trained_model, test_data)

# import data_processing
# import model_training
# import model_evaluation
# import model_testing
# from sklearn.feature_extraction.text import TfidfVectorizer

# if __name__ == "__main__":
#     # Parse training data
#     train_data = data_processing.parse_xml('datasets/train.xml')
#     trial_data = data_processing.parse_xml('datasets/trial.xml')
#     # Combine train and trial datasets
#     combined_train_data = train_data + trial_data
#     # Initialize and fit vectorizer
#     vectorizer = TfidfVectorizer()
#     train_texts = [entry['text'] for entry in combined_train_data]
#     vectorizer.fit(train_texts)
#     # Train model
#     trained_model = model_training.train_model(combined_train_data, vectorizer)
#     # Parse test data
#     test_data = data_processing.parse_xml('datasets/test.xml')
#     print("Number of test samples:", len(test_data))  # Add this line for debugging
#     # Evaluate model
#     model_evaluation.evaluate_model(trained_model, vectorizer, test_data)
#     # Test model
#     model_testing.test_model(trained_model, vectorizer, test_data)

import data_processing
import model_training
import model_evaluation
import model_testing
from sklearn.feature_extraction.text import TfidfVectorizer

if __name__ == "__main__":
    train_data = data_processing.parse_xml('datasets/train.xml')
    if not train_data:
        print("Error: No data found in the training XML file.")
    else:
        trial_data = data_processing.parse_xml('datasets/trial.xml')
        # combined_train_data = train_data + trial_data
        # combined_train_data = train_data + trial_data
        combined_train_data = train_data + trial_data

        print("Number of training samples:", len(combined_train_data))  # Add this line for debugging

        # vectorizer = TfidfVectorizer()
        # train_texts = [sentence['text'] for entry in combined_train_data for sentence in entry.get('sentences', [])]
        # vectorizer.fit(train_texts)

        # if len(vectorizer.vocabulary_) == 0:
        #     print("Error: Empty vocabulary after fitting the vectorizer.")
        # else:
        #     # trained_model, mlb = model_training.train_model(combined_train_data, vectorizer)
        #     # Train models
        #     category_model, polarity_model, target_model, from_model, to_model = model_training.train_models(combined_train_data, vectorizer)
        #     # category_model, polarity_model, target_model, from_model, to_model = model_training.train_models(combined_train_data, vectorizer)

        #     # model_evaluation.evaluate_model(trained_model, vectorizer, combined_train_data, mlb)

        #     test_data = data_processing.parse_xml('datasets/test.xml')
        #     # predictions = model_testing.predict_opinions(trained_model, vectorizer, test_data, mlb)
        #     # Make predictions
        #     predictions = model_testing.predict_opinions(category_model, polarity_model, target_model, from_model, to_model, vectorizer, test_data)

        #     # Print predictions
        #     for i, review in enumerate(test_data):
        #         print(f"Review ID: {review['rid']}")
        #         if i < len(predictions): # Check if predictions exist for the current review
        #             for j, sentence in enumerate(review['sentences']):
        #                 print(f"Sentence {j+1}: {sentence['text']}")
        #                 # Adjust the index calculation to avoid out-of-range errors
        #                 prediction_index = i * len(review['sentences']) + j
        #                 if prediction_index < len(predictions):  # Check if predictions exist for the current sentence
        #                     predicted_opinion = predictions[prediction_index]
        #                     if predicted_opinion:
        #                         print("Predicted Opinion:", predicted_opinion)
        #                     else:
        #                         print("No Opinion available for this sentence.")
        #                 else:
        #                     print("No predictions available for this sentence.")
        #         else:
        #             print("No predictions available for this review.")
        #         print()

        #     # for i, review in enumerate(test_data):
        #     #     print(f"Review ID: {review['rid']}")
        #     #     if i < len(predictions): # Check if predictions exist for the current review
        #     #         for j, sentence in enumerate(review['sentences']):
        #     #             print(f"Sentence {j+1}: {sentence['text']}")
        #     #             # Ensure that the index calculation is correct
        #     #             predicted_opinion = predictions[i * len(review['sentences']) + j]
        #     #             if predicted_opinion:
        #     #                 print("Predicted Opinion:", predicted_opinion)
        #     #             else:
        #     #                 print("No predictions available for this sentence.")
        #     #     else:
        #     #         print("No predictions available for this review.")
        #     #     print()


        #     # model_testing.analyze_predictions(predictions, test_data, mlb)
        #     # print("Analyzed Predictions:")
        #     # print(analyzed_pred)

            
            # for i, review in enumerate(test_data):
            #     print(f"Review ID: {review['rid']}")
            #     if i < len(predictions):  # Check if predictions exist for the current review
            #         for j, sentence in enumerate(review['sentences']):
            #             print(f"Sentence {j+1}: {sentence['text']}")
            #             if j < len(predictions[i]):  # Check if predictions exist for the current sentence
            #                 predicted_opinions = predictions[i][j]
            #                 print(f"Predicted Opinions: {predicted_opinions}")
            #                 if predicted_opinions:
            #                     print("Predicted Opinions:")
            #                     for opinion in predicted_opinions:
            #                         print(f"  Target: {opinion['target']}, Category: {opinion['category']}, Polarity: {opinion['polarity']}, From: {opinion['from']}, To: {opinion['to']}")
            #                 else:
            #                     print("No opinions available for prediction.")
            #             else:
            #                 print("No predictions available for this sentence.")
            #     else:
            #         print("No predictions available for this review.")
            #     print()


        #     # for i, review in enumerate(test_data):
        #     #     print(f"Review ID: {review['rid']}")
        #     #     for j, sentence in enumerate(review['sentences']):
        #     #         print(f"Sentence {j+1}: {sentence['text']}")
        #     #         if predictions is not None:
        #     #             predicted_opinions = predictions[i][j]
        #     #             if predicted_opinions:
        #     #                 print("Predicted Opinions:")
        #     #                 for opinion in predicted_opinions:
        #     #                     print(f"  Target: {opinion['target']}, Category: {opinion['category']}, Polarity: {opinion['polarity']}, From: {opinion['from']}, To: {opinion['to']}")
        #     #             else:
        #     #                 print("No opinions available for prediction.")
        #     #         else:
        #     #             print("Error: No predictions available.")
        #     #     print()


# if __name__ == "__main__":
#     # Parse training data
#     train_data = data_processing.parse_xml('datasets/train.xml')
#     trial_data = data_processing.parse_xml('datasets/trial.xml')
#     # Combine train and trial datasets
#     combined_train_data = train_data + trial_data
    
#     # Initialize vectorizer
#     vectorizer = TfidfVectorizer()
    
#     # Extract texts from combined train data
#     train_texts = []
#     for entry in combined_train_data:
#         if 'text' in entry:
#             train_texts.append(entry['text'])
    
#     if not train_texts:
#         print("Error: No valid texts found in the training data.")
#     else:
#         # Fit vectorizer
#         vectorizer.fit(train_texts)
        
#         if len(vectorizer.vocabulary_) == 0:
#             print("Error: Empty vocabulary after fitting the vectorizer.")
#         else:
#             # Train model
#             trained_model = model_training.train_model(combined_train_data, vectorizer)
            
#             # Evaluate model
#             model_evaluation.evaluate_model(trained_model, vectorizer, combined_train_data)
            
#             # Parse test data
#             test_data = data_processing.parse_xml('datasets/test.xml')
            
#             # Test model
#             predictions = model_testing.predict_opinions(trained_model, vectorizer, test_data)
            
#             # Output predictions
#             for i, review in enumerate(test_data):
#                 print(f"Review ID: {review['rid']}")
#                 for j, sentence in enumerate(review['sentences']):
#                     print(f"Sentence {j+1}: {sentence['text']}")
#                     if predictions is not None:
#                         print(f"Predicted Opinion: {predictions[i][j]}")
#                     else:
#                         print("No opinions available for prediction.")
#                 print()
