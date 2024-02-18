from models.retrieval import get_relevant_segments
from models.llm import format_context
from utils.general_utils import timeit

from constants import SUBSET_TEST_DB_PATH, FITNESS_TEST_QUESTION_1


@timeit
def test_with_subset() -> None:
    '''Tests the retrieval and formatting of relevant segments from the subset database.'''
    
    db_path = SUBSET_TEST_DB_PATH
    
    question = FITNESS_TEST_QUESTION_1
    print("Question: ", question)
    
    # Get the relevant segments from the subset database
    relevant_segments = get_relevant_segments(question,
                   db_path=db_path
                   )
    
    # Format these segments so they are readable
    formatted_segments = format_context(relevant_segments)
    
    print("Segments: ", formatted_segments)
    
if __name__ == "__main__":
    test_with_subset()