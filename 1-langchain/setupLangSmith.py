import os


os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACKING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")
# Set your OpenAI API key
os.environ["OPEN_API_KEY"] = os.getenv("OPEN_API_KEY")

LANGSMITH_TRACING="true"
LANGSMITH_ENDPOINT="https://api.smith.langchain.com"
LANGSMITH_API_KEY= os.getenv("LANGCHAIN_API_KEY")
LANGSMITH_PROJECT= os.getenv("LANGCHAIN_PROJECT")
OPENAI_API_KEY= os.getenv("OPEN_API_KEY")

from langchain_openai import ChatOpenAI

import os
from langchain_openai import ChatOpenAI


# Initialize the model
llm = ChatOpenAI(model="gpt-4o")
print(llm)
