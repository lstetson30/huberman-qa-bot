import openai
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get the API key from the environment variable
api_key = os.getenv("OPENAI_API_KEY")

# Create the OpenAI client
client = openai.OpenAI(api_key=api_key)
    
def answer_question(question, context):
    instruction = '''Answer the users question using the context below. Each blurb starts with a triple backtick.  If there is a relevant blurb that answers the question, reference its "title" and "source" from the JSON below the blurb. If no blurbs are relevant, the source is "N/A". The format should be as follows:\nUser: question\nAI: answer\n[SOURCE: source_url]\n\n''' + str([{'text': result[0], 'metadata': result[2]} for result in context])
    
    response = client.chat.completions.create(
        model = "gpt-3.5-turbo-1106",
        temperature=0.5,
        messages=[
            {"role": "system", "content": instruction},
            {"role": "user", "content": question}
        ]
    )
    
    return response.choices[0].message.content