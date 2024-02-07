import chromadb


def get_relevant_segments(query, db_path, n_results=5):
    client = chromadb.PersistentClient(db_path)
    collection= client.get_collection('huberman_videos')
    
    results = collection.query(query_texts=[query], n_results=n_results)
    
    return results

