from models import llm, retrieval
import gradio as gr


def run_query(question, db_path="data/videos.db", 
              num_rel_segments=5, 
              llm_model="gpt-3.5-turbo-1106", 
              llm_temp=0.5):
    
    relevant_segments = retrieval.get_relevant_segments(question,
                                                         db_path=db_path,
                                                         n_results=num_rel_segments)
    
    answer = llm.answer_with_context(question, 
                                     relevant_segments, 
                                     model=llm_model, 
                                     temperature=llm_temp)
    
    return answer


if __name__ == "__main__":
    demo = gr.Interface(fn=run_query, inputs="text", outputs="text")
    
    demo.launch(share=True)
