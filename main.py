import os
import io
import shutil
import base64
import requests
import json
import traceback
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, UploadFile, File, BackgroundTasks, HTTPException, Form
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List, Optional
from PIL import Image

import database
import document_parser
import video_transcriber

# Lifespan Handler for Startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    database.init_db()
    yield

# Initialize FastAPI App
app = FastAPI(title="Local Ollama RAG", lifespan=lifespan)

# Directories setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(os.path.join(STATIC_DIR, "js"), exist_ok=True)
os.makedirs(os.path.join(STATIC_DIR, "css"), exist_ok=True)
os.makedirs(TEMPLATES_DIR, exist_ok=True)

# Mount Static Files and Templates
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# Models for request validation
class ChatRequest(BaseModel):
    message: str
    session_id: int
    chat_model: str
    embedding_model: str
    doc_ids: Optional[List[int]] = None

class PullRequest(BaseModel):
    model_name: str

# Helper: Ollama URL
OLLAMA_BASE_URL = "http://localhost:11434"

def is_summary_query(text: str) -> bool:
    """Detects if the query is a document summary or overview query."""
    text_lower = text.lower()
    keywords = ["summarize", "summary", "overview", "what is this document about", "what does this document cover", "main points", "key takeaways", "tl;dr", "tldr", "explain the document", "explain this file"]
    return any(kw in text_lower for kw in keywords)

def is_vision_model(model_name: str) -> bool:
    """Detects if the model supports vision capabilities based on its name."""
    name_lower = model_name.lower()
    vision_keywords = ["vision", "llava", "moondream", "minicpm", "bakllava", "qwen-vl", "cogvlm", "multimodal"]
    return any(kw in name_lower for kw in vision_keywords)

def resize_and_compress_image(img_path: str, max_size: int = 1024, quality: int = 80) -> str:
    """Resizes and compresses the image to a smaller JPEG base64 string for faster local vision model processing."""
    try:
        with Image.open(img_path) as img:
            # Convert RGBA/P to RGB (necessary for JPEG)
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")
            
            # Calculate new dimensions preserving aspect ratio
            width, height = img.size
            if max(width, height) > max_size:
                if width > height:
                    new_width = max_size
                    new_height = int(height * (max_size / width))
                else:
                    new_height = max_size
                    new_width = int(width * (max_size / height))
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Save to buffer as JPEG
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=quality)
            return base64.b64encode(buffer.getvalue()).decode("utf-8")
    except Exception as e:
        # Fallback to original file base64 if processing fails
        print(f"Error compressing image {img_path}: {e}")
        with open(img_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode("utf-8")

def get_ollama_embedding(text: str, model: str) -> List[float]:
    """Retrieves text embedding from local Ollama service."""
    # Clean text to prevent transmission issues
    text = text.replace('\x00', '')
    
    # Try /api/embeddings first
    try:
        url = f"{OLLAMA_BASE_URL}/api/embeddings"
        response = requests.post(url, json={"model": model, "prompt": text}, timeout=60)
        if response.status_code == 200:
            return response.json().get("embedding")
        err_msg = ""
        try:
            err_msg = response.json().get("error", "")
        except Exception:
            pass
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Ollama connection error on /api/embeddings: {str(e)}")

    # Fallback to newer /api/embed
    try:
        url_fb = f"{OLLAMA_BASE_URL}/api/embed"
        response_fb = requests.post(url_fb, json={"model": model, "input": text}, timeout=60)
        if response_fb.status_code == 200:
            embeddings = response_fb.json().get("embeddings")
            if embeddings:
                return embeddings[0]
        err_msg_fb = ""
        try:
            err_msg_fb = response_fb.json().get("error", "")
        except Exception:
            pass
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Ollama connection error on /api/embed: {str(e)}")

    # If both failed, decide on the error message to return
    final_err = err_msg_fb or err_msg or f"HTTP {response_fb.status_code}"
    raise HTTPException(status_code=500, detail=f"Ollama embedding failed: {final_err}")

# Background Ingestion Task
def ingest_file_background(doc_id: int, file_path: str, filename: str, file_type: str, embedding_model: str):
    try:
        # Determine ingestion route
        is_media = file_type in ['.mp4', '.avi', '.mkv', '.mov', '.flv', '.webm', '.mp3', '.wav', '.m4a']
        is_image = file_type in ['.png', '.jpg', '.jpeg', '.webp']
        
        chunks = []
        if is_image:
            # Images are parsed as metadata reference, we can chat about them with vision models
            chunks = [{"text": f"Uploaded Image: {filename}. Description / context will be retrieved by multimodal vision LLM.", "page_num": None}]
        elif is_media:
            # Video/Audio files are transcribed via Whisper
            database.update_document_status(doc_id, "transcribing")
            chunks = video_transcriber.process_media_file(file_path, model_name="tiny")
        else:
            # Standard documents
            database.update_document_status(doc_id, "parsing")
            chunks = document_parser.process_file_into_chunks(file_path)
            
        if not chunks:
            raise ValueError("No text or content could be extracted from the file.")
            
        # Generate embeddings and save chunks
        database.update_document_status(doc_id, "indexing")
        for idx, chunk in enumerate(chunks):
            embedding = get_ollama_embedding(chunk["text"], embedding_model)
            database.add_document_chunk(
                doc_id=doc_id,
                text=chunk["text"],
                embedding=embedding,
                page_num=chunk.get("page_num")
            )
            
        database.update_document_status(doc_id, "completed")
    except HTTPException as e:
        tb = traceback.format_exc()
        print(f"Error in background task for doc {doc_id}:\n{tb}")
        database.update_document_status(doc_id, "failed", error_message=e.detail)
    except Exception as e:
        tb = traceback.format_exc()
        print(f"Error in background task for doc {doc_id}:\n{tb}")
        database.update_document_status(doc_id, "failed", error_message=str(e))

# Endpoints
@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.post("/upload")
def upload_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    embedding_model: str = Form("nomic-embed-text")
):
    try:
        filename = file.filename
        file_ext = os.path.splitext(filename)[1].lower()
        file_path = os.path.join(UPLOAD_DIR, filename)
        
        # Save file locally
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Create DB record
        doc_id = database.add_document(filename, file_ext, "queued")
        
        # Dispatch background processing
        background_tasks.add_task(
            ingest_file_background,
            doc_id=doc_id,
            file_path=file_path,
            filename=filename,
            file_type=file_ext,
            embedding_model=embedding_model
        )
        
        return {"status": "success", "document_id": doc_id, "filename": filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents")
def get_documents():
    return database.get_all_documents()

@app.delete("/documents/{doc_id}")
def delete_document(doc_id: int):
    # Retrieve filename to delete the local file
    docs = database.get_all_documents()
    target_doc = next((d for d in docs if d["id"] == doc_id), None)
    if target_doc:
        file_path = os.path.join(UPLOAD_DIR, target_doc["filename"])
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception:
                pass
    database.delete_document(doc_id)
    return {"status": "success"}

@app.get("/ollama/models")
def get_ollama_models():
    """Fetch available models from local Ollama."""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            names = [m["name"] for m in models]
            return {"status": "connected", "models": names}
        return {"status": "disconnected", "error": f"Ollama HTTP {response.status_code}", "models": []}
    except Exception as e:
        return {"status": "disconnected", "error": str(e), "models": []}

@app.post("/ollama/pull")
def pull_ollama_model(req: PullRequest):
    """Pulls a model from Ollama, streaming progress output."""
    def stream_pull():
        try:
            url = f"{OLLAMA_BASE_URL}/api/pull"
            response = requests.post(url, json={"name": req.model_name}, stream=True)
            for line in response.iter_lines():
                if line:
                    yield line.decode("utf-8") + "\n"
        except Exception as e:
            yield json.dumps({"error": f"Failed to pull model: {str(e)}"}) + "\n"
            
    return StreamingResponse(stream_pull(), media_type="text/event-stream")

# Chat Streaming Logic
@app.post("/chat")
def chat_endpoint(req: ChatRequest):
    """Streams a chat conversation incorporating documents RAG."""
    def chat_stream_generator():
        session_id = req.session_id
        message = req.message
        chat_model = req.chat_model
        embedding_model = req.embedding_model
        doc_ids = req.doc_ids
        
        # 1. Store user message in history
        database.add_chat_message(session_id, "user", message)
        
        # 2. Check if we have documents to search
        chunks = []
        try:
            if doc_ids:
                if is_summary_query(message):
                    # For summary/overview queries, retrieve sequential chunks (first 25 chunks)
                    chunks = database.get_document_chunks_sequential(doc_ids, limit=25)
                else:
                    # Retrieve query embedding
                    query_embedding = get_ollama_embedding(message, embedding_model)
                    chunks = database.similarity_search(query_embedding, top_k=5, doc_ids=doc_ids)
                    # Sort semantic results by chunk_id to maintain original text ordering
                    chunks.sort(key=lambda x: x.get("chunk_id", 0))
        except HTTPException as e:
            yield json.dumps({"warning": f"Vector search bypassed or failed: {e.detail}"}) + "\n"
        except Exception as e:
            # We fail gracefully and proceed without document search, logging the error in stream
            yield json.dumps({"warning": f"Vector search bypassed or failed: {str(e)}"}) + "\n"
            
        # 3. Handle image modal input if an image doc is active
        images_base64 = []
        if doc_ids:
            all_docs = database.get_all_documents()
            for doc in all_docs:
                if doc["id"] in doc_ids and doc["file_type"] in ['.png', '.jpg', '.jpeg', '.webp']:
                    img_path = os.path.join(UPLOAD_DIR, doc["filename"])
                    if os.path.exists(img_path):
                        try:
                            b64 = resize_and_compress_image(img_path)
                            images_base64.append(b64)
                        except Exception as e:
                            yield json.dumps({"warning": f"Failed to compress or load image: {str(e)}"}) + "\n"

        # 4. Construct contexts & citations
        context_parts = []
        citations = []
        
        for idx, c in enumerate(chunks):
            source_info = c["filename"]
            if c["page_num"]:
                source_info += f" (Page {c['page_num']})"
            context_parts.append(f"Source [{idx+1}] - {source_info}:\n{c['text']}")
            citations.append({
                "index": idx + 1,
                "filename": c["filename"],
                "page_num": c["page_num"],
                "text": c["text"][:250] + "..."
            })
            
        context_text = "\n\n".join(context_parts)
        
        # 5. Build system prompt
        if chunks:
            system_prompt = (
                "You are a helpful local RAG assistant. Answer the user's question using ONLY the provided context. "
                "If the context does not contain the answer, politely state that you cannot find the answer in the uploaded files. "
                "Do not use external knowledge. For citations, use [1], [2], etc., corresponding to the source indexes.\n\n"
                f"Context:\n{context_text}"
            )
        else:
            system_prompt = (
                "You are a helpful local assistant. Note: No context documents were found or selected. "
                "Answer the user's query normally, but gently let them know they can upload documents, images, "
                "or videos in the sidebar to do local RAG search."
            )
            
        # Add vision warning if images are selected but current model is text-only
        if images_base64 and not is_vision_model(chat_model):
            system_prompt += (
                "\n\nNote: The user has selected one or more image files, but you are a text-only model and cannot see them. "
                "Explicitly and politely inform the user in your response that they must switch to a vision model "
                "(such as moondream or llava) using the 'Chat LLM Model' select dropdown in the left panel to analyze images."
            )
            
        # 6. Build message history (retrieve last 8 messages for context)
        history = database.get_chat_messages(session_id)
        ollama_messages = [{"role": "system", "content": system_prompt}]
        
        # Add past messages, skipping the user message we just saved to avoid duplicates
        for msg in history[-9:-1]:
            ollama_messages.append({"role": msg["role"], "content": msg["content"]})
            
        # Add the current user query (with optional base64 images if using a vision model)
        user_msg = {"role": "user", "content": message}
        if images_base64 and is_vision_model(chat_model):
            user_msg["images"] = images_base64
        ollama_messages.append(user_msg)
        
        # 7. Request Ollama API chat completion
        assistant_content = []
        try:
            url = f"{OLLAMA_BASE_URL}/api/chat"
            response = requests.post(url, json={
                "model": chat_model,
                "messages": ollama_messages,
                "stream": True,
                "options": {
                    "num_ctx": 16384,      # Increase context window to 16k tokens
                    "num_predict": 16384     # Increase max output generation to 2048 tokens
                }
            }, stream=True, timeout=180)
            
            if response.status_code != 200:
                yield json.dumps({"error": f"Ollama HTTP {response.status_code}: {response.text}"}) + "\n"
                return
                
            for line in response.iter_lines():
                if line:
                    data = json.loads(line.decode("utf-8"))
                    token = data.get("message", {}).get("content", "")
                    if token:
                        assistant_content.append(token)
                        yield json.dumps({"token": token}) + "\n"
                        
            # Save final response to Database
            full_response = "".join(assistant_content)
            database.add_chat_message(session_id, "assistant", full_response, citations)
            
            # Send citations metadata to frontend
            yield json.dumps({"citations": citations}) + "\n"
            
        except Exception as e:
            yield json.dumps({"error": f"Ollama communication failure: {str(e)}"}) + "\n"

    return StreamingResponse(chat_stream_generator(), media_type="text/event-stream")

# Chat Sessions endpoints
@app.post("/sessions")
def create_session():
    session_id = database.create_chat_session()
    return {"session_id": session_id}

@app.get("/sessions")
def get_sessions():
    return database.get_chat_sessions()

@app.delete("/sessions/{session_id}")
def delete_session(session_id: int):
    database.delete_chat_session(session_id)
    return {"status": "success"}

@app.get("/sessions/{session_id}/messages")
def get_messages(session_id: int):
    return database.get_chat_messages(session_id)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
