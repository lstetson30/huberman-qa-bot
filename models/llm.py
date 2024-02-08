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
    
    instruction = '''Answer the user's question using the RELEVANT CONTEXT provided by the user, if possible. If there is a CONTEXT that seems to answer the question, structure you answer around that context and return its TITLE and SOURCE. If no CONTEXTs are relevant to the question, answer the question yourself and state that no relevant clips were found. The format should be as follows:\n
    User: ```What is muscle atrophy?```\n
    AI: ```Muscle atrophy is the decrease in size and wasting of muscle tissue.\n
    [TITLE] title from relevant CONTEXT\n
    [SOURCE] url from relevant CONTEXT```
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