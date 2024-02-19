---
title: Fitness_QA_Bot
emoji: 💪
colorFrom: purple
colorTo: purple
app_file: main.py
sdk: gradio
sdk_version: 4.16.0
pinned: true
license: mit
---

# Fitness Q&A
This project is a simple question-answering (Q&A) bot focused on fitness-related queries. It utilized a combination of machine learning models and retrieval techniques to provide informative responses to user questions.

Check out the chatbot here: https://huggingface.co/spaces/lstetson/Fitness_QA_Bot

The main objective is to assist users in obtaining relevant information about fitness and recovery topics. The bot accepts user questions as inputs and returns answers along with a link to Dr. Andrew Huberman's videos for further context. The videos come from his [Fitness and Recovery playlist](https://www.youtube.com/playlist?list=PLPNW_gerXa4O24l7ZHoJbMC2xOO7SpS7K).

## Usage
To run the project, you need to have Python installed on your system along with the required dependencies specified in ``requirements.txt``.

```bash
pip install -r requirements.txt
```

After installing the dependencies, you need to set up your OpenAI API Key. You can sign up for an API Key at [OpenAI's website](https://openai.com/). Once you have your API Key, you should set it as an environment variable named ``OPENAI_API_KEY``.
```bash
export OPENAI_API_KEY='your-api-key'
```

You can now run the `main.py` file. This will launch a Gradio interface where you can interact with the system.

## Data Extraction, Transformation, and Loading (ETL)

In addition to the main functionality provided by `main.py`, this project includes a script for Data Extraction, Transformation, and Loading (ETL). This script, `run_etl.py`, allows you to extract metadata (YouTube ids and video titles) from a JSON file, extract data, transform it, and load it into a database.

### Usage

To use the ETL script, follow these steps:

1. Navigate to the root directory of the project in your terminal.
2. Run the `run_etl.py` script using Python:

```bash
python run_etl.py
```
You will be prompted to provide the following information:

1. Path to the JSON file containing the data.
2. Path to the database where you want to store the transformed data.
3. Batch size for processing the data (leave blank for no batching).
4. Batch Overlap (leave blank for no overlap).

Here's an example:

Enter the path to the JSON file: **data/input_data.json**  
Enter the path to the database: **data/output_database.db**  
Enter batch size (leave blank for no batching): **10**  
Enter overlap (leave blank for no overlap): **2**  

## Next Steps/Improvements
* Evaluate: Create question/expected-answer pairs and compare model outputs
* Tune: Optimize hyperparameters (num relevant segments, LLM temp, etc.)
* Get human feedback

## Acknowledgements
* Dr. Andrew Huberman - For his informative videos on health-related topics.
* [The Full Stack](https://github.com/the-full-stack/) - For inspiring this project and providing helpful resources.