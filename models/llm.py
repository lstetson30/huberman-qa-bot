import openai
import os
from dotenv import load_dotenv

from constants import DEFAULT_LLM_MODEL, DEFAULT_LLM_TEMP

# Load environment variables from .env file
load_dotenv()

# Get the API key from the environment variable
api_key = os.getenv("OPENAI_API_KEY")

# Create the OpenAI client
client = openai.OpenAI(api_key=api_key)

# Create base to provide context to LLM
context_result_base = "CONTEXT: {text}\nTITLE: {title}\nSOURCE: {source}\n\n"

LLM_INSTRUCTION = """You are a question-answering bot. The user will ask a question about fitness and recovery. First, you will be provided relevant context. The relevant context are segments of transcripts from Andrew Huberman's playlist on fitness and recovery where he has conversations about these topics. Answer the user's question and include the video title and link to the relevant context where they talk about the topic of the user's question.  When referencing relevant context, return its TITLE and SOURCE. If no context are related to the question, answer the question yourself and state that "No relevant clips were found". Use this format:
User: ```What is muscle atrophy?```
AI: ```Muscle atrophy is the decrease in size and wasting of muscle tissue.
VIDEO: Example video title
SOURCE: Example video url```
    """

def format_context(db_query_results: dict) -> str:
    """Formats the context for the LLM response.

    Args:
        db_query_results (dict): The results of the database query.

    Returns:
        str: The formatted context for the LLM response.
    """

    # Get the documents and metadatas from the database query results
    documents = db_query_results["documents"][0]
    metadatas = db_query_results["metadatas"][0]

    # Initialize the formatted context string
    formatted_context = ""

    # Loop through the documents and metadatas
    for i in range(len(documents)):
        # Format the single context result
        result_text = context_result_base.format(
            text=documents[i],
            title=metadatas[i]["title"],
            source=metadatas[i]["source"],
        )

        # Add this context result to the formatted context string
        formatted_context += result_text

    return formatted_context


def answer_with_context(
    question: str,
    context: dict,
    model: str = DEFAULT_LLM_MODEL,
    temperature: float = DEFAULT_LLM_TEMP,
) -> str:
    f"""Answers the user's question using the LLM model and relevant context.

    Args:
        question (str): The user's question.
        context (dict): The relevant context from the database query.
        model (str): The LLM model to use. Default is {DEFAULT_LLM_MODEL}.
        temperature (float): The sampling temperature to use. Default is {DEFAULT_LLM_TEMP}.

    Returns:
        str: The LLM response to the user's question.
    """

    # Format the context so it can be read by the LLM
    formatted_context = format_context(context)

    # Provide instruction to the LLM
    instruction = LLM_INSTRUCTION

    # Add RELEVANT CONTEXT heading for LLM
    formatted_context = "RELEVANT CONTEXT:\n```" + formatted_context + "```"

    # Get LLM response by providing instruction, context, and question
    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": instruction},
            {"role": "user", "content": formatted_context},
            {"role": "user", "content": question},
        ],
    )

    return response.choices[0].message.content
