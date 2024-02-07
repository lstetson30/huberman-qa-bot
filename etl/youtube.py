import json
import torch
import sqlite3
import numpy as np
import io

from utils import timeit
from youtube_transcript_api import YouTubeTranscriptApi
from transformers import AutoTokenizer, AutoModel


@timeit
def main(json_path="data/videos.json", db=None, batches=False, batch_size=10):
    with open(json_path) as f:
        video_info = json.load(f)
        
    videos = []
    for video in video_info:
        video_id = video["id"]
        video_title = video["title"]
        transcript = get_video_transcript(video_id)
        print(f"Transcript for video {video_id} fetched.")
        if transcript:
            if batches:
                formatted_transcript = format_transcript_in_batches(transcript, video_id, video_title, batch_size=batch_size)
            else:
                formatted_transcript = format_transcript(transcript, video_id, video_title)
            transcript_with_embeddings = add_embeddings_to_data(formatted_transcript)
            print(f"Embeddings added for video {video_id}.")
            
            videos.extend(transcript_with_embeddings)
    
    if db:
        load_data_to_db(db, videos)
    else:
        print("No database specified. Skipping database load.")
        print(videos)

def get_video_transcript(video_id):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en', 'en-US'])
        return transcript
    except Exception as e:
        print(f"Error fetching transcript for video {video_id}: {str(e)}")
        return None
    
    
def format_transcript(transcript, video_id, video_title):
    formatted_data = []
    base_url = f"https://www.youtube.com/watch?v={video_id}"
    query_params = "&t={start}s"
    
    for i, entry in enumerate(transcript, start=1):
        url = base_url + query_params.format(start=entry["start"])
        
        metadata = {
            "video_id": video_id,
            "segment_id": i,
            "title": video_title,
            "source": url
        }
        
        segment = {"text": entry["text"], "metadata": metadata}
        
        formatted_data.append(segment)

    return formatted_data


def format_transcript_in_batches(transcript, video_id, video_title, batch_size=10, overlap=3):
    formatted_data = []
    base_url = f"https://www.youtube.com/watch?v={video_id}"
    query_params = "&t={start}s"
    
    for i in range(0, len(transcript), batch_size - overlap):
        batch = transcript[i:i+batch_size]
        
        start_time = batch[0]["start"]
        
        text = " ".join(entry["text"] for entry in batch)
        
        url = base_url + query_params.format(start=start_time)
        
        metadata = {
            "video_id": video_id,
            "segment_id": i+1,
            "title": video_title,
            "source": url
        }
        
        segment = {"text": text, "metadata": metadata}
        
        formatted_data.append(segment)

    return formatted_data


model_name = "YituTech/conv-bert-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def embed_text(text):
    inputs = tokenizer(text, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy()
    return embeddings


def add_embeddings_to_data(transcript):
    for segment in transcript:
        segment['embeddings'] = embed_text(segment['text'])
    return transcript


def load_data_to_db(db_path, data):
    def adapt_array(arr):
        """
        http://stackoverflow.com/a/31312102/190597 (SoulNibbler)
        """
        out = io.BytesIO()
        np.save(out, arr)
        out.seek(0)
        return sqlite3.Binary(out.read())
    
    # Converts np.array to TEXT when inserting
    sqlite3.register_adapter(np.ndarray, adapt_array)
    
    conn = sqlite3.connect(db_path, detect_types=sqlite3.PARSE_DECLTYPES)
    c = conn.cursor()

    c.execute('''
        CREATE TABLE IF NOT EXISTS videos (
            text TEXT,
            embedding array,
            metadata TEXT
        )
    ''')

    for segment in data:
        text = segment['text']
        embedding = segment['embeddings']
        metadata = json.dumps(segment['metadata'])

        c.execute('''
            INSERT OR IGNORE INTO videos (text, embedding, metadata) VALUES (?, ?, ?)
            ''', (text, embedding, metadata))

    conn.commit()
    conn.close()
    
    print(f"Data loaded to database at {db_path}.")
    
    
if __name__ == "__main__":
    json_path = input("Enter path to video JSON file: ")
    db = input("Enter path to database file (leave blank to skip): ")
    batches = input("Process transcript in batches? (y/n):")
    
    if batches.lower() == 'y':
        main(json_path, db, batches=True, batch_size=5)
    elif batches.lower() == 'n':
        main(json_path, db, batches=False)
    else:
        print("Invalid input. Exiting.")