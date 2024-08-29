import os
import json
import numpy as np
import faiss
import re
import spacy
from datasets import load_dataset
from bs4 import BeautifulSoup
from urllib.parse import unquote  # Import unquote for decoding URL-encoded titles
# Load the spaCy model
# nlp = spacy.load('en_core_web_sm')

# def segment_sentences(text):
#     # Process the text with spaCy
#     doc = nlp(text)
    
#     # Extract sentences
#     sentences = [sent.text.strip() for sent in doc.sents]
    
#     return sentences

# # Example text
# text = "Dr. Smith went to Washington. He visited the U.S. Capitol. It was a great trip."

# # Segment sentences
# segmented_sentences = segment_sentences(text)

# print(segmented_sentences)

def preprocess_text(text):
    # Remove HTML tags
    text = BeautifulSoup(text, "html.parser").get_text()
     # Handle escaped quotes and other escape sequences properly
    text = text.encode('utf-8').decode('unicode_escape')
    # Remove newline characters and unnecessary spaces
    text = re.sub(r'\n+', ' ', text)  # Replace newlines with a space
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    return text

def clean_text(text):
    # Define the unwanted characters
    unwanted_prefix = "\u00c3\u00af\u00c2\u00bb\u00c2\u00bf"
    if text.startswith(unwanted_prefix):
        return text[len(unwanted_prefix):]
    return text

def create_concatenated_documents_squad_json(output_json_path='data/squad/concatenated_documents.json', max_titles=10):
    # Load the SQuAD dataset (SQuAD v1.1 in this case)
    dataset = load_dataset('squad', split='train')

    # Initialize variables
    documents = []
    unique_titles = set()
    document_id = 1  # Start document IDs from 1
    title_qas_count = {}
    total_qas_count = 0
    
    # Iterate over the dataset to find the first `max_titles` unique titles and concatenate their unique contexts
    for example in dataset:
        # Ensure that the example is a dictionary and contains the 'title' and 'context' keys
        if isinstance(example, dict) and 'title' in example and 'context' in example:
            title = unquote(example['title'])
            
            if title not in unique_titles:
                unique_titles.add(title)
                document_entry = {
                    'id': document_id,
                    'title': title,
                    'content': [],
                    'qas': []
                }
                documents.append(document_entry)
                document_id += 1  # Increment document ID
                title_qas_count[title] = 0  # Initialize Q&A count for this title

            context = example['context']

            # Find the document entry with the matching title
            for doc in documents:
                if doc['title'] == title:
                    # Add context if it's not already in the content list
                    if context not in doc['content']:
                        doc['content'].append(context)
                    qas_entry = {
                        'question': (example['question']),
                        'context': context,
                        'answers': example['answers']['text']
                    }
                    doc['qas'].append(qas_entry)
                    title_qas_count[title] += 1  # Increment Q&A count for this title
                    total_qas_count += 1  # Increment total Q&A count
            
            # Stop when we have `max_titles` unique titles
            if len(unique_titles) >= max_titles:
                break

    # Concatenate the contexts for each document
    for doc in documents:
        doc['content'] = "\n".join(doc['content'])
        doc['content'] = preprocess_text(doc['content'])

    # Print the number of Q&As per title and the total Q&A count
    for title, count in title_qas_count.items():
        print(f"Title: {title}, Q&A Count: {count}")
    print(f"Total Q&A Count: {total_qas_count}")

    # Create the 'data' directory if it doesn't exist
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)

    # Write the documents to a JSON file
    with open(output_json_path, 'w', encoding='utf-8') as json_file:
        json.dump(documents, json_file, ensure_ascii=False, indent=4)

    print(f"Concatenated contexts with IDs have been saved to {output_json_path}")

def create_concatenated_documents_narrativeqa_json(output_json_path='data/narrativeqa/concatenated_documents.json', max_titles=10):
    # Load the NarrativeQA dataset (train split)
    dataset = load_dataset('deepmind/narrativeqa', split='train')
    
    # Initialize variables
    documents_dict = {}
    unique_titles = set()
    total_qas_count = 0
    # Iterate through the dataset
    for item in dataset:
        document_id = item['document']['id']
        document_title = item['document']['summary']['title']
        
        # Check if the title is unique and if we have collected less than max_titles
        if document_title not in unique_titles and len(documents_dict) < max_titles:
            unique_titles.add(document_title)
        
        # Aggregate questions and answers for each unique document ID
        if document_id not in documents_dict:
            documents_dict[document_id] = {
                "summary_title": document_title,
                "qa_pairs": [],
                "document_text": clean_text(preprocess_text(item['document']['text'])),
            }
        
        # Create a question-answer pair
        qa_pair = {
            "question": item['question']['text'],
            "answers": [answer['text'] for answer in item['answers']]
        }
        
        # Add the question-answer pair to the list
        documents_dict[document_id]["qa_pairs"].append(qa_pair)

        # Increment the total_qas counter
        total_qas_count += 1
        # Stop once we have max_titles unique titles
        if len(unique_titles) == max_titles:
            break

    # Convert the documents_dict to a list for saving to JSON
    unique_documents = list(documents_dict.values())

    # Create the 'data' directory if it doesn't exist
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)

    # Save the aggregated documents to a JSON file
    with open(output_json_path, 'w', encoding='utf-8') as outfile:
        json.dump(unique_documents, outfile, indent=4)

    print(f"Saved {len(unique_documents)} unique documents to {output_json_path}")
    print(f"Total number of question-answer pairs (qas): {total_qas_count}")
create_concatenated_documents_narrativeqa_json()