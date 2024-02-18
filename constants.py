MAIN_VIDEOS_JSON_PATH="data/videos.json"
MAIN_VIDEOS_DB_PATH="data/videos.db"
SINGLE_VIDEO_TEST_DB_PATH="data/single_video.db"
SUBSET_TEST_DB_PATH="data/videos_subset_more_context.db"

EMBEDDING_MODEL="YituTech/conv-bert-base"

TABLE_NAME="huberman_videos"
DISTANCE_METRIC="cosine"

DEFAULT_LLM_MODEL="gpt-3.5-turbo-0125"
DEFAULT_LLM_TEMP=0.1

DEFAULT_QUERY_RESULTS=5

LLM_TEST_QUESTION="What are the components of an LLM?"
FITNESS_TEST_QUESTION="How should I train for anerobic capacity?"
FITNESS_TEST_QUESTION_1="What methods can I use to lose weight quickly?"

GRADIO_TITLE="Fitness Q&A"
GRADIO_DESCRIPTION="Ask me a question about fitness! If I can, I'll provide a link to one of Dr. Andrew Huberman's videos with more information."
GRADIO_EXAMPLES=["How can I promote muscle recovery?", "How does caffeine affect my workout?"]