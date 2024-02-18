import chromadb

from constants import TABLE_NAME, DEFAULT_QUERY_RESULTS


def get_relevant_segments(query, db_path, n_results=DEFAULT_QUERY_RESULTS):
    client = chromadb.PersistentClient(db_path)
    collection= client.get_collection(TABLE_NAME)
    
    results = collection.query(query_texts=[query], n_results=n_results)
    
    return results

