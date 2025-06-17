from mcp.server.fastmcp import FastMCP
from pg_memory_system import PGMemorySystem
from typing import List, Optional
import os

# Initialize FastMCP instance
mcp = FastMCP("memory-mcp-sse")

# Initialize memory system
memory_system = PGMemorySystem(
    model_name='all-MiniLM-L6-v2',
    llm_backend="openai",
    llm_model="qwen-max-latest",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1/",
    db_url=os.getenv('DATABASE_URL')
)

@mcp.tool()
def add_memory(content: str, tags: Optional[List[str]] = None, category: Optional[str] = None) -> int:
    """
    Add a new memory to the system.

    Parameters:
    - content (str): The memory content text (required)
    - tags (List[str]): List of tags for the memory (optional)
    - category (str): Category of the memory (optional)

    Returns:
    - int: ID of the inserted memory
    """
    return memory_system.add_memory(content, tags, category)

@mcp.tool()
def search_memories(query: str, k: int = 5) -> List[dict]:
    """
    Search for similar memories using vector similarity.

    Parameters:
    - query (str): Search query text (required)
    - k (int): Number of results to return (optional, default=5)

    Returns:
    - List[dict]: List of memory dictionaries with content and metadata
    """
    return memory_system.search_memories(query, k)

@mcp.tool()
def get_memory(memory_id: int) -> dict:
    """
    Get a memory by ID.

    Parameters:
    - memory_id (int): ID of the memory to retrieve (required)

    Returns:
    - dict: Memory dictionary with content and metadata
    """
    return memory_system.get_memory(memory_id)

@mcp.tool()
def update_memory(memory_id: int, content: Optional[str] = None, tags: Optional[List[str]] = None, category: Optional[str] = None) -> bool:
    """
    Update a memory.

    Parameters:
    - memory_id (int): ID of the memory to update (required)
    - content (str): New content text (optional)
    - tags (List[str]): New tags list (optional)
    - category (str): New category (optional)

    Returns:
    - bool: True if update successful
    """
    return memory_system.update_memory(memory_id, content, tags, category)

@mcp.tool()
def delete_memory(memory_id: int) -> bool:
    """
    Delete a memory.

    Parameters:
    - memory_id (int): ID of the memory to delete (required)

    Returns:
    - bool: True if deletion successful
    """
    return memory_system.delete_memory(memory_id) 