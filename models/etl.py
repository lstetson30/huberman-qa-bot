import json
import sqlite3
import numpy as np
import io
import chromadb

from utils.general_utils import timeit
from utils.embedding_utils import MyEmbeddingFunction
from youtube_transcript_api import YouTubeTranscriptApi


@timeit
def main(json_path="data/videos.json", db=None, batch_size=None, overlap=None):
    with open(json_path) as f:
        video_info = json.load(f)
        
    videos = []
    for video in video_info:
        video_id = video["id"]
        video_title = video["title"]
        transcript = get_video_transcript(video_id)
        print(f"Transcript for video {video_id} fetched.")
        if transcript:
            formatted_transcript = format_transcript(transcript, video_id, video_title, batch_size=batch_size, overlap=overlap)
            
            videos.extend(formatted_transcript)
    
    if db:
        initialize_db(db)
        load_data_to_db(db, videos)
    else:
        print("No database specified. Skipping database load.")
        print(videos)
        

@timeit
def get_video_transcript(video_id):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en', 'en-US'])
        return transcript
    except Exception as e:
        print(f"Error fetching transcript for video {video_id}: {str(e)}")
        return None
    
    
def format_transcript(transcript, video_id, video_title, batch_size=None, overlap=None):
    formatted_data = []
    base_url = f"https://www.youtube.com/watch?v={video_id}"
    query_params = "&t={start}s"
    
    if not batch_size:
        batch_size = 1
        overlap = 0
        
    for i in range(0, len(transcript), batch_size - overlap):
        batch = list(transcript[i:i+batch_size])
        
        start_time = batch[0]["start"]
        
        text = " ".join(entry["text"] for entry in batch)
        
        url = base_url + query_params.format(start=start_time)
        
        metadata = {
            "video_id": video_id,
            "segment_id": video_id + "__" + str(i),
            "title": video_title,
            "source": url
        }
        
        segment = {"text": text, "metadata": metadata}
        
        formatted_data.append(segment)

    return formatted_data


embed_text = MyEmbeddingFunction()

def initialize_db(db_path, distance_metric="cosine"):
    client = chromadb.PersistentClient(path=db_path)
    
    # Clear existing data
    # client.reset()
    
    client.create_collection(
        name="huberman_videos",
        embedding_function=embed_text,
        metadata={"hnsw:space": distance_metric}
    )
    
    print(f"Database created at {db_path}")
    

def load_data_to_db(db_path, data):
    client = chromadb.PersistentClient(path=db_path)
    
    collection = client.get_collection("huberman_videos")

    documents = [segment['text'] for segment in data]
    metadata = [segment['metadata'] for segment in data]
    ids = [segment['metadata']['segment_id'] for segment in data]

    collection.add(
        documents=documents,
        metadatas=metadata,
        ids=ids
    )

    print(f"Data loaded to database at {db_path}.")
    
    