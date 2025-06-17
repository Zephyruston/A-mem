import os
import json
import logging
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime
import psycopg2
from psycopg2.extras import Json
from sentence_transformers import SentenceTransformer
from llm_controller import LLMController

logger = logging.getLogger(__name__)

class PGMemorySystem:
    """PostgreSQL-based memory system using pgvector for vector storage."""
    
    def __init__(self, 
                 model_name: str = 'all-MiniLM-L6-v2',
                 llm_backend: str = "openai",
                 llm_model: str = "gpt-4",
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None,
                 db_url: Optional[str] = None):
        """Initialize the PostgreSQL memory system.
        
        Args:
            model_name: Name of the sentence transformer model
            llm_backend: LLM backend to use (openai/ollama)
            llm_model: Name of the LLM model
            api_key: API key for the LLM service
            base_url: Base URL for the LLM service
            db_url: PostgreSQL connection URL
        """
        self.model_name = model_name
        self.embedding_model = SentenceTransformer(model_name)
        self.llm_controller = LLMController(llm_backend, llm_model, api_key, base_url)
        
        # Initialize database connection
        self.db_url = db_url or os.getenv('DATABASE_URL')
        if not self.db_url:
            raise ValueError("Database URL not provided. Set DATABASE_URL environment variable or pass db_url parameter.")
            
        # Create table if not exists
        self._init_db()
        
    def _init_db(self):
        """Initialize database table."""
        with psycopg2.connect(self.db_url) as conn:
            with conn.cursor() as cur:
                # Enable pgvector extension
                cur.execute('CREATE EXTENSION IF NOT EXISTS vector;')
                
                # Create vector_store table
                cur.execute('''
                    CREATE TABLE IF NOT EXISTS vector_store (
                        id SERIAL PRIMARY KEY,
                        timestamp TIMESTAMP(6),
                        text TEXT,
                        embedding VECTOR(384),
                        model VARCHAR(50),
                        metadata JSONB
                    );
                ''')
                
                # Create index for vector similarity search
                cur.execute('''
                    CREATE INDEX IF NOT EXISTS vector_store_embedding_idx 
                    ON vector_store 
                    USING ivfflat (embedding vector_cosine_ops)
                    WITH (lists = 100);
                ''')
                
    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding vector for text."""
        return self.embedding_model.encode(text).tolist()
        
    def add_memory(self, 
                  content: str,
                  tags: Optional[List[str]] = None,
                  category: Optional[str] = None,
                  timestamp: Optional[str] = None) -> int:
        """Add a new memory to the system.
        
        Args:
            content: Memory content text
            tags: List of tags
            category: Memory category
            timestamp: Timestamp in YYYYMMDDHHmm format
            
        Returns:
            int: ID of the inserted memory
        """
        # Generate embedding
        embedding = self._get_embedding(content)
        
        # Prepare metadata
        metadata = {
            'tags': tags or [],
            'category': category or 'Uncategorized',
            'timestamp': timestamp or datetime.now().strftime('%Y%m%d%H%M')
        }
        
        # Insert into database
        with psycopg2.connect(self.db_url) as conn:
            with conn.cursor() as cur:
                cur.execute('''
                    INSERT INTO vector_store (timestamp, text, embedding, model, metadata)
                    VALUES (%s, %s, %s, %s, %s)
                    RETURNING id;
                ''', (
                    datetime.strptime(metadata['timestamp'], '%Y%m%d%H%M'),
                    content,
                    embedding,
                    self.model_name,
                    Json(metadata)
                ))
                memory_id = cur.fetchone()[0]
                
        return memory_id
        
    def search_memories(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar memories using vector similarity.
        
        Args:
            query: Search query text
            k: Number of results to return
            
        Returns:
            List of memory dictionaries with content and metadata
        """
        query_embedding = self._get_embedding(query)
        
        with psycopg2.connect(self.db_url) as conn:
            with conn.cursor() as cur:
                cur.execute('''
                    SELECT id, text, metadata, 1 - (embedding <=> %s::vector) as similarity
                    FROM vector_store
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s;
                ''', (query_embedding, query_embedding, k))
                
                results = []
                for row in cur.fetchall():
                    results.append({
                        'id': row[0],
                        'content': row[1],
                        'metadata': row[2],
                        'similarity': row[3]
                    })
                    
        return results
        
    def get_memory(self, memory_id: int) -> Optional[Dict[str, Any]]:
        """Get a memory by ID.
        
        Args:
            memory_id: Memory ID
            
        Returns:
            Memory dictionary if found, None otherwise
        """
        with psycopg2.connect(self.db_url) as conn:
            with conn.cursor() as cur:
                cur.execute('''
                    SELECT id, text, metadata
                    FROM vector_store
                    WHERE id = %s;
                ''', (memory_id,))
                
                row = cur.fetchone()
                if row:
                    return {
                        'id': row[0],
                        'content': row[1],
                        'metadata': row[2]
                    }
                return None
                
    def update_memory(self, 
                     memory_id: int,
                     content: Optional[str] = None,
                     tags: Optional[List[str]] = None,
                     category: Optional[str] = None) -> bool:
        """Update a memory.
        
        Args:
            memory_id: Memory ID
            content: New content text
            tags: New tags list
            category: New category
            
        Returns:
            bool: True if update successful
        """
        # Get existing memory
        memory = self.get_memory(memory_id)
        if not memory:
            return False
            
        # Update fields
        if content:
            memory['content'] = content
            embedding = self._get_embedding(content)
        else:
            embedding = None
            
        if tags:
            memory['metadata']['tags'] = tags
        if category:
            memory['metadata']['category'] = category
            
        # Update in database
        with psycopg2.connect(self.db_url) as conn:
            with conn.cursor() as cur:
                if embedding:
                    cur.execute('''
                        UPDATE vector_store
                        SET text = %s, embedding = %s, metadata = %s
                        WHERE id = %s;
                    ''', (memory['content'], embedding, Json(memory['metadata']), memory_id))
                else:
                    cur.execute('''
                        UPDATE vector_store
                        SET metadata = %s
                        WHERE id = %s;
                    ''', (Json(memory['metadata']), memory_id))
                    
        return True
        
    def delete_memory(self, memory_id: int) -> bool:
        """Delete a memory.
        
        Args:
            memory_id: Memory ID
            
        Returns:
            bool: True if deletion successful
        """
        with psycopg2.connect(self.db_url) as conn:
            with conn.cursor() as cur:
                cur.execute('DELETE FROM vector_store WHERE id = %s;', (memory_id,))
                return cur.rowcount > 0 