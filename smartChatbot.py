import pdfplumber
import pinecone
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

# Initialize Pinecone
pinecone.init(api_key="your-pinecone-api-key", environment="your-environment")
index_name = "pdf-chatbot"

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF using pdfplumber."""
    documents = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                documents.append(Document(page_content=text, metadata={"page": page_num}))
    return documents

# Load the PDF
pdf_path = "your_document.pdf"
documents = extract_text_from_pdf(pdf_path)

# Split the extracted text into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_docs = text_splitter.split_documents(documents)

# Convert text into embeddings
embedding_model = OpenAIEmbeddings()

# Store embeddings in Pinecone vector database
vector_store = Pinecone.from_documents(split_docs, embedding_model, index_name=index_name)

# Create a retriever using cosine similarity
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Implement RAG (Retrieval-Augmented Generation)
llm = OpenAI(model_name="gpt-4")
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# Function to query the chatbot
def ask_question(query):
    response = qa_chain({"query": query})
    print("Answer:", response["result"])
    for doc in response["source_documents"]:
        print("\nRetrieved Document:", doc.page_content[:500])  # Display first 500 characters

# Example query
if __name__ == "__main__":
    query = input("Ask a question: ")
    ask_question(query)
