import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import model_training, model_testing, data_processing
from time import sleep
import tqdm


# data processing loading bar
def sleep_loading_bar(text):
    for i in tqdm.tqdm(range(100)):
        sleep(0.05)
    print(text)



def main():

    # Parse the data
    xml_files = ['datasets/train.xml', 'datasets/trial.xml', 'datasets/test.xml']
    total_xml_files = len(xml_files)
    # loading bar 
    for i in range(total_xml_files):
        print(f"Loading XML file {i+1}/{total_xml_files}...")
        # sleep_loading_bar(f"Loading XML file {i+1}/{total_xml_files}...")


    xml_files = [data_processing.parse_xml(file) for file in xml_files]
    total_parsed_files = len(xml_files)
    # loading bar
    for i in range(total_parsed_files):
        sleep_loading_bar(f"XML file {i+1}/{total_parsed_files} parsed successfully.")

            
    combined_train_data = xml_files[0] + xml_files[1]
    test_data = xml_files[2]

    # combined_data = train_data + trial_data
    # print("Total number of reviews in the combined dataset:", len(combined_data))

    print("Total number of reviews in the combined dataset:", len(combined_train_data))
    print("Total number of reviews in the test dataset:", len(test_data))


    # Initialize and fit the TfidfVectorizer
    vectorizer = TfidfVectorizer()
    train_texts = [sentence['text'] for entry in combined_train_data for sentence in entry.get('sentences', [])]
    vectorizer.fit(train_texts)

    sleep_loading_bar("Training models.")
    # Train the models
    clf_category, clf_polarity, clf_target, clf_from, clf_to = model_training.train_model(combined_train_data, vectorizer)

    # loading bar 
    # Save the trained models using pickle
    with open('model/category_model.pkl', 'wb') as f:
        pickle.dump(clf_category, f)
    with open('model/polarity_model.pkl', 'wb') as f:
        pickle.dump(clf_polarity, f)
    with open('model/target_model.pkl', 'wb') as f:
        pickle.dump(clf_target, f)
    with open('model/from_model.pkl', 'wb') as f:
        pickle.dump(clf_from, f)
    with open('model/to_model.pkl', 'wb') as f:
        pickle.dump(clf_to, f)
    

    sleep_loading_bar("Models trained and saved successfully.")
    sleep_loading_bar("Loading the models...")

    # Load the trained models from pickle
    with open('model/category_model.pkl', 'rb') as f:
        clf_category = pickle.load(f)
    with open('model/polarity_model.pkl', 'rb') as f:
        clf_polarity = pickle.load(f)
    with open('model/target_model.pkl', 'rb') as f:
        clf_target = pickle.load(f)
    with open('model/from_model.pkl', 'rb') as f:
        clf_from = pickle.load(f)
    with open('model/to_model.pkl', 'rb') as f:
        clf_to = pickle.load(f)

    sleep_loading_bar("Models loaded successfully.")
    sleep_loading_bar("Making predictions on the test data...")



    # Make predictions on the test data
    predictions = model_testing.predict_opinions(clf_category, clf_polarity, clf_target, clf_from, clf_to, vectorizer, test_data)
    # loading bar
    sleep_loading_bar("Predictions made successfully.")

    # Print predictions
    for review in test_data:
        print(f"Review ID: {review['rid']}")
        for sentence in review['sentences']:
            print(f"Sentence: {sentence['text']}")
            # Find the prediction corresponding to the current sentence
            predicted_opinion = None
            for prediction in predictions:
                if prediction['sentence'][0] == sentence['id']:  # Compare the first element of the tuple
                    predicted_opinion = prediction['opinion']
                    break

            # Print the predicted opinion
            if predicted_opinion:
                print("Predicted Opinion:", predicted_opinion)
            else:
                print("No predictions available for this sentence.")
        print()


    # total reviews predicted
    print("Total reviews predicted:", len(predictions))
    # how many test datasets was predicted
    print("Total test datasets predicted:", len(test_data))



if __name__ == "__main__":
    main()
