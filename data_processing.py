# import xml.etree.ElementTree as ET

# def parse_xml(file_path):
#     # Parse XML file and extract necessary attributes
#     tree = ET.parse(file_path)
#     root = tree.getroot()
#     data = []
#     for review in root.findall('Review'):
#         for sentence in review.find('sentences'):
#             text = sentence.find('text').text
#             opinions = sentence.find('Opinions')
#             if opinions is not None:
#                 for opinion in opinions.findall('Opinion'):
#                     target = opinion.get('target')
#                     category = opinion.get('category')
#                     polarity = opinion.get('polarity')
#                     start_idx = int(opinion.get('from'))
#                     end_idx = int(opinion.get('to'))
#                     data.append({'text': text, 'target': target, 'category': category, 'polarity': polarity, 'start_idx': start_idx, 'end_idx': end_idx})
#     return data

# def preprocess_data(data):
#     # Implement preprocessing steps such as tokenization, removing stop words, etc.
#     # Here we'll just return the data as is for demonstration
#     return data

# import xml.etree.ElementTree as ET

# def parse_xml(file_path):
#     """
#     Parses XML file containing review data and returns a list of dictionaries,
#     each representing a review with its sentences and opinions.
#     """
#     reviews = []
#     tree = ET.parse(file_path)
#     root = tree.getroot()
#     for review in root.findall('Review'):
#         review_data = {'rid': review.attrib['rid'], 'sentences': []}
#         for sentence in review.find('sentences').findall('sentence'):
#             sentence_data = {'id': sentence.attrib['id'], 'text': sentence.find('text').text}
#             opinions = sentence.find('Opinions')
#             if opinions is not None:
#                 sentence_data['opinions'] = opinions
#             else:
#                 sentence_data['opinions'] = None
#             review_data['sentences'].append(sentence_data)
#         reviews.append(review_data)
#     return reviews
# data_processing.py
import xml.etree.ElementTree as ET

def parse_xml(file_path):
    """
    Parses XML file containing review data and returns a list of dictionaries,
    each representing a review with its sentences and opinions.
    """
    reviews = []
    tree = ET.parse(file_path)
    root = tree.getroot()
    for review in root.findall('Review'):
        review_data = {'rid': review.attrib['rid'], 'sentences': []}
        for sentence in review.find('sentences').findall('sentence'):
            sentence_data = {'id': sentence.attrib['id'], 'text': sentence.find('text').text}
            opinions = sentence.find('Opinions')
            if opinions is not None:
                opinions_list = []
                for opinion in opinions.findall('Opinion'):
                    opinion_data = {'target': opinion.get('target', 'NULL'),
                                    'category': opinion.get('category', ''),
                                    'polarity': opinion.get('polarity', ''),
                                    'from': int(opinion.get('from', '0')),  # Convert 'from' attribute to int
                                    'to': int(opinion.get('to', '0'))}      # Convert 'to' attribute to int
                    opinions_list.append(opinion_data)
                sentence_data['opinions'] = opinions_list
            else:
                # Handle case where no opinions are found
                sentence_data['opinions'] = []
                # Set 'from' attribute to 0 when target is 'NULL'
                if sentence_data['text'] and all(opinion['target'] == 'NULL' for opinion in sentence_data['opinions']):
                    sentence_data['opinions'] = [{'target': 'NULL', 'category': '', 'polarity': '', 'from': 0, 'to': 0}]
            review_data['sentences'].append(sentence_data)
            
            # Calculate 'from' and 'to' attributes if target word is not NULL
            if sentence_data['opinions'][0]['target'] != 'NULL':
                target_word = sentence_data['opinions'][0]['target']
                from_position, to_position = calculate_from_to(sentence_data['text'], target_word)
                sentence_data['opinions'][0]['from'] = from_position
                sentence_data['opinions'][0]['to'] = to_position
                
        reviews.append(review_data)
    return reviews


def calculate_from_to(sentence, target_word):
    start_index = sentence.find(target_word)
    end_index = start_index + len(target_word) - 1
    
    return start_index, end_index