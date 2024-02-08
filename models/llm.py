import openai
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get the API key from the environment variable
api_key = os.getenv("OPENAI_API_KEY")

# Create the OpenAI client
client = openai.OpenAI(api_key=api_key)

context_result_base = "CONTEXT: {text}\nTITLE: {title}\nSOURCE: {source}\n\n"

def format_context(db_query_results):
    documents = db_query_results['documents'][0]
    metadatas = db_query_results['metadatas'][0]
    
    formatted_context = ""
    for i in range(len(documents)):
        result_text = context_result_base.format(text=documents[i], title=metadatas[i]['title'], source=metadatas[i]['source'])
        formatted_context += result_text
    
    return formatted_context


def answer_with_context(question, context, model="gpt-3.5-turbo-1106", temperature=0.5):
    formatted_context = format_context(context)
    
    instruction = '''You are a question-answering bot. The user will ask a question about fitness and recovery. First, you will be provided relevant context. The relevant context are segments of transcripts from Andrew Huberman's playlist on fitness and recovery where he has conversations about these topics. Answer the user's question and include the video title and link to the relevant context where they talk about the topic of the user's question.  When referencing relevant context, return its TITLE and SOURCE. If no context are related to the question, answer the question yourself and state that "No relevant clips were found". Use this format:
User: ```What is muscle atrophy?```
AI: ```Muscle atrophy is the decrease in size and wasting of muscle tissue.
VIDEO: Example video title
SOURCE: Example video url```
    '''
    
    formatted_context = "RELEVANT CONTEXT:\n```" + formatted_context + "```"
    
    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": instruction},
            {"role": "user", "content": formatted_context},
            {"role": "user", "content": question}
        ]
    )
    
    return response.choices[0].message.content