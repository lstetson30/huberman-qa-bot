from models import llm, retrieval
import pprint
import gradio as gr

def a(question):
    source_data = retrieval.get_data_from_db("data/single_video.db")
    
    relevant_segments = retrieval.search_segments_with_embedding(question, source_data)
    
    answer = llm.answer_question(question, relevant_segments)
    
    return answer


if __name__ == "__main__":
    demo = gr.Interface(fn=a, inputs="text", outputs="text")
    
    demo.launch(share=True)

# if __name__ == "__main__":
#     source_data = retrieval.get_data_from_db("data/single_video.db")
    
#     query = input("Enter a question: ")
    
#     relevant_segments = retrieval.search_segments_with_embedding(query, source_data)
    
#     pp = pprint.PrettyPrinter()
#     pp.pprint(llm.answer_question(query, relevant_segments))