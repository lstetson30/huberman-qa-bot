from models import llm, retrieval
import gradio as gr

from constants import MAIN_VIDEOS_DB_PATH, DEFAULT_QUERY_RESULTS, DEFAULT_LLM_MODEL, DEFAULT_LLM_TEMP
from constants import GRADIO_TITLE, GRADIO_DESCRIPTION, GRADIO_EXAMPLES


def run_query(question, db_path=MAIN_VIDEOS_DB_PATH,
              num_rel_segments=DEFAULT_QUERY_RESULTS, 
              llm_model=DEFAULT_LLM_MODEL, 
              llm_temp=DEFAULT_LLM_TEMP):
    
    relevant_segments = retrieval.get_relevant_segments(question,
                                                         db_path=db_path,
                                                         n_results=num_rel_segments)
    
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
