from main import run_query
from utils.general_utils import timeit

from constants import (
    MAIN_VIDEOS_DB_PATH,
    SINGLE_VIDEO_TEST_DB_PATH,
    SUBSET_TEST_DB_PATH,
    LLM_TEST_QUESTION,
    FITNESS_TEST_QUESTION,
)


@timeit
def test_with_llm_video() -> None:
    """Test the main function with a single video database."""

    db_path = SINGLE_VIDEO_TEST_DB_PATH

    question = LLM_TEST_QUESTION
    print("Question: ", question)

    # Run the query using the single video database
    answer = run_query(question, db_path=db_path)

    print("Answer: ", answer)


@timeit
def test_with_subset() -> None:
    """Test the main function with a subset database."""

    db_path = SUBSET_TEST_DB_PATH

    question = FITNESS_TEST_QUESTION
    print("Question: ", question)

    # Run the query using the subset of the main database
    answer = run_query(question, db_path=db_path)

    print("Answer: ", answer)


@timeit
def test_with_fullset() -> None:
    """Test the main function with the full database."""

    db_path = MAIN_VIDEOS_DB_PATH

    question = FITNESS_TEST_QUESTION
    print("Question: ", question)

    # Run the query using the main database
    answer = run_query(question, db_path=db_path)

    print("Answer: ", answer)


if __name__ == "__main__":
    choice = input(
        "Test with:\n1. fullset\n2. subset\n3. LLM video\nEnter option number: "
    )

    if choice == "1":
        test_with_fullset()
    elif choice == "2":
        test_with_subset()
    elif choice == "3":
        test_with_llm_video()
    else:
        print("Invalid choice")
        exit(1)
