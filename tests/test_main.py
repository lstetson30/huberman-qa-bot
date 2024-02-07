from main import run_query
from utils.general_utils import timeit

@timeit
def test_with_llm_video():
    db_path = "data/single_video.db"
    
    question = "What are the components of an LLM?"
    print("Question: ", question)
    
    answer = run_query(question,
                   db_path=db_path
                   )
    
    print("Answer: ", answer)


if __name__ == '__main__':
    test_with_llm_video()