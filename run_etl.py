from models.etl import run_etl

if __name__ == "__main__":
    json_path = input("Enter path to JSON file: ")
    db_path = input("Enter path to database: ")
    batch_size = int(input("Enter batch size (leave blank for no batching): "))

    if batch_size:
        overlap = int(input("Enter overlap (leave blank for no overlap): "))
    else:
        overlap = None

    run_etl(json_path=json_path, db=db_path, batch_size=batch_size, overlap=overlap)
