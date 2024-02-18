import chromadb

from constants import TABLE_NAME, DEFAULT_QUERY_RESULTS


def get_relevant_segments(
    query: str, db_path: str, n_results: int = DEFAULT_QUERY_RESULTS
) -> dict:
    f"""Gets the relevant segments from the database for the user's query.

    Args:
        query (str): The user's query.
        db_path (str): The path to the database file.
        n_results (int): The number of results to return. Default is {DEFAULT_QUERY_RESULTS}.

    Returns:
        dict: The relevant segments from the database.
    """

    # Access the database client and collection (table)
    client = chromadb.PersistentClient(db_path)
    collection = client.get_collection(TABLE_NAME)

    # Query the database for the relevant segments
    results = collection.query(query_texts=[query], n_results=n_results)

    return results
