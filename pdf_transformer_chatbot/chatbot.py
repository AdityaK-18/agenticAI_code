import os
from langchain_groq import ChatGroq
from similarity_search import similarity_search

# ✅ Set up Groq API Key
os.environ["GROQ_API_KEY"] = "your_actual_groq_api_key"

# ✅ Initialize LLM (Groq API)
llm = ChatGroq(model="mixtral-8x7b-32768")

# ✅ Function to generate chatbot response
def chatbot(query):
    relevant_text = similarity_search.retrieve(query)
    prompt = f"Context: {relevant_text}\nUser: {query}\nBot:"
    response = llm.invoke(prompt)
    return response.content.strip()
