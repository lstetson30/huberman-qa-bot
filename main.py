from models import llm, retrieval
import gradio as gr

from constants import MAIN_VIDEOS_DB_PATH, DEFAULT_QUERY_RESULTS, DEFAULT_LLM_MODEL, DEFAULT_LLM_TEMP
from constants import GRADIO_TITLE, GRADIO_DESCRIPTION, GRADIO_EXAMPLES


def run_query(question:str, db_path:str=MAIN_VIDEOS_DB_PATH,
              num_rel_segments:int=DEFAULT_QUERY_RESULTS, 
              llm_model:str=DEFAULT_LLM_MODEL, 
              llm_temp:float=DEFAULT_LLM_TEMP) -> str:
    f'''Runs the query and returns the answer.
    
    Args:
        question (str): The user's question.
        db_path (str): The path to the database. Default is {MAIN_VIDEOS_DB_PATH}.
        num_rel_segments (int): The number of relevant segments to retrieve. Default is {DEFAULT_QUERY_RESULTS}.
        llm_model (str): The LLM model to use. Default is {DEFAULT_LLM_MODEL}.
        llm_temp (float): The sampling temperature to use. Default is {DEFAULT_LLM_TEMP}.
        
    Returns:
        str: The answer to the user's question.
    '''
    
    # Get the relevant segments
    relevant_segments = retrieval.get_relevant_segments(question,
                                                         db_path=db_path,
                                                         n_results=num_rel_segments)
    
    # Get the LLM answer
    answer = llm.answer_with_context(question, 
                                     relevant_segments, 
                                     model=llm_model, 
                                     temperature=llm_temp)
    
    return answer


if __name__ == "__main__":
    demo = gr.Interface(fn=run_query, 
                        inputs="text", 
                        outputs="text",
                        title=GRADIO_TITLE,
                        description=GRADIO_DESCRIPTION,
                        theme="soft",
                        examples=GRADIO_EXAMPLES
                       )
    
    demo.launch(share=True)
