from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from pdf_processor import extract_text_from_pdf, split_text_into_chunks
from similarity_search import similarity_search
from chatbot import chatbot

app = FastAPI()

class Query(BaseModel):
    text: str

# ✅ API endpoint to chat with the chatbot
@app.post("/chat")
def chat(query: Query):
    response = chatbot(query.text)
    return {"response": response}

# ✅ API endpoint to upload and process PDFs
@app.post("/upload_pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    content = await file.read()

    # Save file temporarily
    pdf_path = f"temp_{file.filename}"
    with open(pdf_path, "wb") as f:
        f.write(content)

    # Process PDF
    text = extract_text_from_pdf(pdf_path)
    chunks = split_text_into_chunks(text)

    # Build FAISS index
    similarity_search.build_index(chunks)

    return {"message": f"PDF {file.filename} processed successfully!"}

# ✅ Run API
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
