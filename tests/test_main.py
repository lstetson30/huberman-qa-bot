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

@timeit
def test_with_subset():
    db_path = "data/videos_subset.db"
    
    question = "What methods can I use to lose weight quickly?"
    print("Question: ", question)
    
    answer = run_query(question,
                   db_path=db_path
                   )
    
    print("Answer: ", answer)


if __name__ == '__main__':
    choice = input("Enter 1 for test_with_subset, 2 for test_with_llm_video: ")
    
    if choice == "1":
        test_with_subset()
    elif choice == "2":
        test_with_llm_video()
    else:
        print("Invalid choice")
        exit(1)