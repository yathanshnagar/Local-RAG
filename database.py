import sqlite3
import json
import os
import numpy as np
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "rag_store.db")

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Documents table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS documents (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT NOT NULL,
        file_type TEXT NOT NULL,
        upload_time TEXT NOT NULL,
        status TEXT NOT NULL,
        error_message TEXT
    )
    """)
    
    # Document chunks table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS document_chunks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        document_id INTEGER NOT NULL,
        text TEXT NOT NULL,
        embedding TEXT NOT NULL,  -- JSON serialized list of floats
        page_num INTEGER,
        FOREIGN KEY (document_id) REFERENCES documents (id) ON DELETE CASCADE
    )
    """)
    
    # Chat sessions table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS chat_sessions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        created_time TEXT NOT NULL
    )
    """)
    
    # Chat messages table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS chat_messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id INTEGER NOT NULL,
        role TEXT NOT NULL,       -- 'user' or 'assistant'
        content TEXT NOT NULL,
        citations TEXT,           -- JSON serialized list of source citations
        timestamp TEXT NOT NULL,
        FOREIGN KEY (session_id) REFERENCES chat_sessions (id) ON DELETE CASCADE
    )
    """)
    
    conn.commit()
    conn.close()

# Document Operations
def add_document(filename, file_type, status="processing"):
    conn = get_db_connection()
    cursor = conn.cursor()
    now = datetime.now().isoformat()
    cursor.execute(
        "INSERT INTO documents (filename, file_type, upload_time, status) VALUES (?, ?, ?, ?)",
        (filename, file_type, now, status)
    )
    doc_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return doc_id

def update_document_status(doc_id, status, error_message=None):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE documents SET status = ?, error_message = ? WHERE id = ?",
        (status, error_message, doc_id)
    )
    conn.commit()
    conn.close()

def add_document_chunk(doc_id, text, embedding, page_num=None):
    conn = get_db_connection()
    cursor = conn.cursor()
    embedding_json = json.dumps(embedding)
    cursor.execute(
        "INSERT INTO document_chunks (document_id, text, embedding, page_num) VALUES (?, ?, ?, ?)",
        (doc_id, text, embedding_json, page_num)
    )
    conn.commit()
    conn.close()

def delete_document(doc_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    # SQLite does foreign key cascades, but let's make sure it deletes chunks too
    cursor.execute("DELETE FROM document_chunks WHERE document_id = ?", (doc_id,))
    cursor.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
    conn.commit()
    conn.close()

def get_all_documents():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM documents ORDER BY id DESC")
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def get_document_chunks_sequential(doc_ids, limit=25):
    """Retrieves document chunks in sequential order from start to finish up to a limit."""
    if not doc_ids:
        return []
    conn = get_db_connection()
    cursor = conn.cursor()
    placeholders = ",".join("?" for _ in doc_ids)
    query = f"""
        SELECT c.id, c.document_id, c.text, c.page_num, d.filename 
        FROM document_chunks c 
        JOIN documents d ON c.document_id = d.id 
        WHERE c.document_id IN ({placeholders})
        ORDER BY c.document_id ASC, c.id ASC
        LIMIT ?
    """
    cursor.execute(query, list(doc_ids) + [limit])
    rows = cursor.fetchall()
    conn.close()
    
    results = []
    for row in rows:
        results.append({
            "chunk_id": row['id'],
            "document_id": row['document_id'],
            "filename": row['filename'],
            "text": row['text'],
            "page_num": row['page_num'],
            "score": 1.0
        })
    return results

# Similarity Search using NumPy
def similarity_search(query_embedding, top_k=5, doc_ids=None):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Query chunks from database
    if doc_ids:
        placeholders = ",".join("?" for _ in doc_ids)
        query = f"""
            SELECT c.id, c.document_id, c.text, c.embedding, c.page_num, d.filename 
            FROM document_chunks c 
            JOIN documents d ON c.document_id = d.id 
            WHERE c.document_id IN ({placeholders})
        """
        cursor.execute(query, doc_ids)
    else:
        query = """
            SELECT c.id, c.document_id, c.text, c.embedding, c.page_num, d.filename 
            FROM document_chunks c 
            JOIN documents d ON c.document_id = d.id
        """
        cursor.execute(query)
        
    rows = cursor.fetchall()
    conn.close()
    
    if not rows:
        return []
        
    # Calculate cosine similarity in Python using NumPy
    scores = []
    query_vec = np.array(query_embedding, dtype=np.float32)
    query_norm = np.linalg.norm(query_vec)
    
    if query_norm == 0:
        return []
        
    for row in rows:
        try:
            chunk_vec = np.array(json.loads(row['embedding']), dtype=np.float32)
            chunk_norm = np.linalg.norm(chunk_vec)
            if chunk_norm == 0:
                similarity = 0.0
            else:
                similarity = float(np.dot(query_vec, chunk_vec) / (query_norm * chunk_norm))
            
            scores.append({
                "chunk_id": row['id'],
                "document_id": row['document_id'],
                "filename": row['filename'],
                "text": row['text'],
                "page_num": row['page_num'],
                "score": similarity
            })
        except Exception as e:
            # Skip corrupted embeddings
            continue
            
    # Sort by score descending and return top_k
    scores.sort(key=lambda x: x['score'], reverse=True)
    return scores[:top_k]

# Chat History Operations
def create_chat_session(title=None):
    conn = get_db_connection()
    cursor = conn.cursor()
    now = datetime.now().isoformat()
    if not title:
        title = f"Chat at {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    cursor.execute(
        "INSERT INTO chat_sessions (title, created_time) VALUES (?, ?)",
        (title, now)
    )
    session_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return session_id

def get_chat_sessions():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM chat_sessions ORDER BY id DESC")
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def delete_chat_session(session_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM chat_messages WHERE session_id = ?", (session_id,))
    cursor.execute("DELETE FROM chat_sessions WHERE id = ?", (session_id,))
    conn.commit()
    conn.close()

def add_chat_message(session_id, role, content, citations=None):
    conn = get_db_connection()
    cursor = conn.cursor()
    now = datetime.now().isoformat()
    citations_json = json.dumps(citations) if citations else None
    cursor.execute(
        "INSERT INTO chat_messages (session_id, role, content, citations, timestamp) VALUES (?, ?, ?, ?, ?)",
        (session_id, role, content, citations_json, now)
    )
    conn.commit()
    conn.close()

def get_chat_messages(session_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM chat_messages WHERE session_id = ? ORDER BY id ASC", (session_id,))
    rows = cursor.fetchall()
    conn.close()
    
    messages = []
    for row in rows:
        msg = dict(row)
        if msg['citations']:
            msg['citations'] = json.loads(msg['citations'])
        messages.append(msg)
    return messages
