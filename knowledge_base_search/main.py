from fastapi import FastAPI, UploadFile, File
import fitz  # PyMuPDF to read PDFs
from PIL import Image
import io
from sentence_transformers import SentenceTransformer
import nltk
import numpy as np
from pydantic import BaseModel

app = FastAPI()
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings_store = []

# Download NLTK punkt tokenizer (run once automatically)
nltk.download('punkt')

local_pipe = None  # global variable for lazy loading

def get_local_pipe():
    global local_pipe
    if local_pipe is None:
        from transformers import pipeline
        local_pipe = pipeline("text-generation", model="google/flan-t5-base", max_new_tokens=256)
    return local_pipe

class QueryModel(BaseModel):
    query: str

@app.post("/upload/")
async def upload_document(file: UploadFile = File(...)):
    contents = await file.read()
    doc = fitz.open(stream=contents, filetype="pdf")
    extracted_text = ""
    images = []

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        extracted_text += page.get_text()
        for img in page.get_images():
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            images.append(image_bytes)  # Store binary image data

    from nltk.tokenize import sent_tokenize
    sentences = sent_tokenize(extracted_text)

    for sentence in sentences:
        if sentence.strip():
            emb = model.encode(sentence)
            embeddings_store.append({
                "text": sentence,
                "embedding": emb.tolist()
            })

    return {
        "filename": file.filename,
        "text_excerpt": extracted_text[:500],
        "sentence_count": len(sentences),
        "image_count": len(images)
    }

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

@app.post("/search/")
async def search_document(data: QueryModel):
    query = data.query
    if not query:
        return {"error": "No query provided"}

    query_embedding = model.encode(query)
    similarities = [
        (cosine_similarity(query_embedding, item["embedding"]), item["text"])
        for item in embeddings_store
    ]
    similarities.sort(reverse=True, key=lambda x: x[0])
    top_matches = similarities[:3]

    return {
        "query": query,
        "top_matches": top_matches
    }

@app.post("/synthesize/")
async def synthesize_answer(data: QueryModel):
    query = data.query
    query_embedding = model.encode(query)
    similarities = [
        (cosine_similarity(query_embedding, item["embedding"]), item["text"])
        for item in embeddings_store
    ]
    similarities.sort(reverse=True, key=lambda x: x[0])
    top_chunks = [t[1] for t in similarities[:3]]
    context = "\n".join(top_chunks)
    prompt = (
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer using only the context above, as clearly as possible."
    )
    pipe = get_local_pipe()
    result = pipe(prompt)[0]['generated_text']
    return {
        "question": query,
        "context": top_chunks,
        "answer": result
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
