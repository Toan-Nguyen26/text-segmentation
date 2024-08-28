import faiss
import json
from sentence_transformers import SentenceTransformer

# Load your FAISS index
index = faiss.read_index('data/faiss_index.index')

# Load the document metadata (e.g., original texts or chunk info)
with open('data/document_ids.json', 'r', encoding='utf-8') as f:
    document_store = json.load(f)

# Load the model for encoding queries
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def encode_query(query):
    query_embedding = model.encode([query])
    return query_embedding

def search_faiss_index(query_embedding, top_k=5):
    # Perform the search on FAISS index
    D, I = index.search(query_embedding, top_k)
    return I  # Return the indices of the top chunks

def get_top_chunks(indices):
    top_chunks = []
    for idx in indices[0]:
        # Assuming document_store contains the relevant chunk text and metadata
        chunk_info = {
            'id': document_store[idx]['id'],
            'doc_id': document_store[idx]['doc_id'],
            'chunk_text': document_store[idx]['chunk'],
            'embedding': document_store[idx]['embedding']
        }
        top_chunks.append(chunk_info)
    return top_chunks

def ask_question_and_retrieve_chunks(question):
    query_embedding = encode_query(question)
    indices = search_faiss_index(query_embedding, top_k=5)
    top_chunks = get_top_chunks(indices)
    return top_chunks

if __name__ == "__main__":
    question = "How many undergrads were attending Notre Dame in 2014?"
    top_chunks = ask_question_and_retrieve_chunks(question)
    
    # Prepare the results to be saved
    results = []
    for i, chunk in enumerate(top_chunks):
        result = {
            'rank': i + 1,
            'id': chunk['id'],
            'document_id': chunk['doc_id'],
            'text': chunk['chunk_text']
        }
        results.append(result)
    
    # Save the results to a JSON file
    output_file = 'retrieved_chunks.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    
    print(f"Results have been saved to {output_file}")