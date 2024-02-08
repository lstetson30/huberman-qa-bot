from models import llm, retrieval
import gradio as gr


def run_query(question, db_path="data/videos_subset_more_context.db", #dev path
              num_rel_segments=5, 
              llm_model="gpt-3.5-turbo-0125", 
              llm_temp=0.1):
    
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
                        title="Fitness Q&A",
                        description="Ask me a question about fitness! If I can, I'll provide a link to one of Dr. Andrew Huberman's videos with more information.",
                        theme="soft",
                        examples=["How can I promote muscle recovery?", "How does caffeine affect my workout?"]
                       )
    
    demo.launch(share=True)
