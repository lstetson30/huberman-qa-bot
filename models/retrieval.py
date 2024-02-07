from sklearn.metrics.pairwise import cosine_similarity
from ..utils.embedding_utils import MyEmbeddingFunction
import numpy as np
import io
import sqlite3


def get_data_from_db(db_path):
    def convert_array(text):
        out = io.BytesIO(text)
        out.seek(0)
        return np.load(out)
    
    # Converts TEXT to np.array when selecting
    sqlite3.register_converter("array", convert_array)
    
    conn = sqlite3.connect(db_path, detect_types=sqlite3.PARSE_DECLTYPES)
    cursor = conn.cursor()

    cursor.execute('''
        SELECT *
        FROM videos
    ''')
    
    data = cursor.fetchall()
    
    conn.close()
    
    return data


def get_most_similar_indexes(query_embedding, segment_embeddings, top_k=10):
    similarities = cosine_similarity([query_embedding], segment_embeddings)
    most_similar_indexes = np.argsort(similarities)[0][-top_k:][::-1]
    return most_similar_indexes


embed_text = MyEmbeddingFunction()


def search_segments_with_embedding(query, source_data):
    # Embed user's query
    query_embedding = embed_text(query)

    # Embed transcript segments
    segment_embeddings = [segment[1] for segment in source_data]

    # Find the most similar segment
    most_similar_indexes = get_most_similar_indexes(query_embedding, segment_embeddings)

    return [source_data[index] for index in most_similar_indexes]

if __name__ == "__main__":
    source_data = get_data_from_db("data/single_video.db")
    
    query = input("Enter a question: ")
    
    relevant_segments = search_segments_with_embedding(query, source_data)
    
    for segment in relevant_segments:
        print(segment[0])
        print(segment[2])
        print()