import json
import chromadb
from datetime import datetime
import math
from typing import List, Dict

from utils.general_utils import timeit
from utils.embedding_utils import MyEmbeddingFunction
from youtube_transcript_api import YouTubeTranscriptApi

from constants import MAIN_VIDEOS_JSON_PATH, TABLE_NAME, DISTANCE_METRIC


@timeit
def get_video_transcript(video_id: str) -> List[Dict[str, str, str]]:
    """Fetches the transcript for a given YouTube video ID using the YouTubeTranscriptApi.

    Args:
        video_id (str): The YouTube video ID.

    Returns:
        list: A list of dictionaries containing the transcript for the video.
                {'text': 'The text of the segment', 'start': 'The start time of the segment', 'duration': 'The duration of the segment'}
    """

    try:
        transcript = YouTubeTranscriptApi.get_transcript(
            video_id, languages=["en", "en-US"]
        )
        return transcript
    except Exception as e:
        print(f"Error fetching transcript for video {video_id}: {str(e)}")
        return None


def format_transcript(
    transcript: list,
    video_id: str,
    video_title: str,
    batch_size: int = None,
    overlap: int = None,
) -> List[Dict[str, Dict[str, str, str, str]]]:
    """Formats the transcript into segments for loading into the database. Metadata is added to each segment to include the video ID, segment ID, title, and source URL. Batch size and overlap can be specified to create overlapping segments.

    Args:
        transcript (list): The transcript for the video.
        video_id (str): The YouTube video ID.
        video_title (str): The title of the video.
        batch_size (int): The number of segments to include in each batch. Default is None.
        overlap (int): The number of overlapping segments between each batch. Default is None.

    Returns:
        list: A list of dictionaries containing the formatted segments.
                {'text': 'The text of the segment',
                'metadata':
                    {'video_id': 'The YouTube video ID',
                    'segment_id': 'The segment ID',
                    'title': 'The title of the video',
                    'source': 'The source URL'
                    }
                }
    """

    # Initialize the list to store the formatted segments
    formatted_data = []
    # Specify base YouTube URL and timestamp query parameter
    base_url = f"https://www.youtube.com/watch?v={video_id}"
    query_params = "&t={start}s"

    # If no batching, loop through each segment
    if not batch_size:
        batch_size = 1
        overlap = 0

    # Loop through the transcript in batches
    for i in range(0, len(transcript), batch_size - overlap):
        # Get the batch of segments
        batch = list(transcript[i : i + batch_size])

        # Start time is the start time of the first segment
        start_time = batch[0]["start"]

        # Join all of the text from the segments in the batch
        text = " ".join(entry["text"] for entry in batch)

        # Set the URL for the start of the batch
        url = base_url + query_params.format(start=start_time)

        # Set metadata for the batch
        metadata = {
            "video_id": video_id,
            "segment_id": video_id + "__" + str(i),
            "title": video_title,
            "source": url,
        }

        segment = {"text": text, "metadata": metadata}

        # Add this batch to the formatted data
        formatted_data.append(segment)

    return formatted_data


# Initialize the embedding function
embed_text = MyEmbeddingFunction()


def initialize_db(db_path: str, distance_metric: str = DISTANCE_METRIC) -> None:
    f"""Initializes the database with the specified distance metric.

    Args:
        db_path (str): The path to the database file.
        distance_metric (str): The distance metric to use for the database. Default is {DISTANCE_METRIC}.
    """

    # Create a persistent database client
    client = chromadb.PersistentClient(path=db_path)

    # Create a table using the embedding function and specified distance metric
    client.create_collection(
        name=TABLE_NAME,
        embedding_function=embed_text,
        metadata={"hnsw:space": distance_metric},
    )

    print(f"Database created at {db_path}")


def load_data_to_db(db_path: str, data: list) -> None:
    """Loads the formatted data into the database.

    Args:
        db_path (str): The path to the database file.
        data (list): A list of dictionaries containing the formatted segments.
                {'text': 'The text of the segment',
                'metadata':
                    {'video_id': 'The YouTube video ID',
                    'segment_id': 'The segment ID',
                    'title': 'The title of the video',
                    'source': 'The source URL'
                    }
                }
    """
    # Access the database client
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_collection(TABLE_NAME)

    # Load the data in batches. ChromaDB has a limit of 5461 documents per batch.
    num_rows = len(data)
    batch_size = 5461
    num_batches = math.ceil(num_rows / batch_size)

    for i in range(num_batches):
        batch_data = data[i * batch_size : (i + 1) * batch_size]
        documents = [segment["text"] for segment in batch_data]
        metadata = [segment["metadata"] for segment in batch_data]
        ids = [segment["metadata"]["segment_id"] for segment in batch_data]

        collection.add(documents=documents, metadatas=metadata, ids=ids)
        print(f"Batch {i+1} of {num_batches} loaded to database.")

    print(f"Data loaded to database at {db_path}.")


def log_data_load(json_path: str, db_path: str, batch_size: int, overlap: int) -> None:
    """Logs the data load to a JSON file.

    Args:
        json_path (str): The path to the JSON file containing the video information.
        db_path (str): The path to the database file.
        batch_size (int): The number of segments to include in each batch.
        overlap (int): The number of overlapping segments between each batch.
    """

    # Create a JSON string with the data load information
    log_json = json.dumps(
        {
            "videos_info_path": json_path,
            "db_path": db_path,
            "batch_size": batch_size,
            "overlap": overlap,
            "load_time": str(datetime.now()),
        }
    )

    # Get the database name from the path and set the log path
    db_file = db_path.split("/")[-1]
    db_name = db_file.split(".")[0]
    log_path = f"data/logs/{db_name}_load_log.json"

    # Write the log to a JSON file
    with open(log_path, "w") as f:
        f.write(log_json)


@timeit
def run_etl(
    json_path: str = MAIN_VIDEOS_JSON_PATH,
    db: str = None,
    batch_size: int = None,
    overlap: int = None,
) -> None:
    f"""Runs the ETL process to fetch video transcripts, format the data, and load it into the database.

    Args:
        json_path (str): The path to the JSON file containing the video information. Default is {MAIN_VIDEOS_JSON_PATH}.
        db (str): The path to the database file. Default is None.
        batch_size (int): The number of segments to include in each batch. Default is None.
        overlap (int): The number of overlapping segments between each batch. Default is None.
    """

    # Load the video information from the JSON file
    with open(json_path) as f:
        video_info = json.load(f)

    # Initialize the list to store the formatted segments
    videos = []

    # Loop through the video information and fetch the transcript for each video
    for video in video_info:
        video_id = video["id"]
        video_title = video["title"]
        transcript = get_video_transcript(video_id)
        print(f"Transcript for video {video_id} fetched.")

        # If the transcript is fetched, format the data and add it to the list
        if transcript:
            formatted_transcript = format_transcript(
                transcript,
                video_id,
                video_title,
                batch_size=batch_size,
                overlap=overlap,
            )

            videos.extend(formatted_transcript)

    # If a database is specified, initialize the database, load the data, and log the data load
    if db:
        initialize_db(db)
        load_data_to_db(db, videos)
        log_data_load(json_path, db, batch_size, overlap)
    else:
        print("No database specified. Skipping database load.")
        print(videos)  # Print the formatted data for troubleshooting
