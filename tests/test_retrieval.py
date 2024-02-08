from models.retrieval import get_relevant_segments
from models.llm import format_context
from utils.general_utils import timeit

@timeit
def test_with_subset():
    db_path = "data/videos_subset.db"
    
    question = "What methods can I use to lose weight quickly?"
    print("Question: ", question)
    
    relevant_segments = get_relevant_segments(question,
                   db_path=db_path
                   )
    formatted_segments = format_context(relevant_segments)
    
    print("Segments: ", formatted_segments)
    
if __name__ == "__main__":
    test_with_subset()