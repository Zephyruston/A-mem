import argparse
from mcp.server.fastmcp import FastMCP
from starlette.applications import Starlette
from mcp.server.sse import SseServerTransport
from starlette.requests import Request
from starlette.routing import Mount, Route
from mcp.server import Server
import logging
import uvicorn
from pg_memory_system import PGMemorySystem
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("memory-mcp-sse")

# Initialize memory system
memory_system = PGMemorySystem(
    model_name='all-MiniLM-L6-v2',
    llm_backend="openai",
    llm_model="qwen-max-latest",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1/",
    db_url=os.getenv('DATABASE_URL')
)

# Initialize FastMCP instance
mcp = FastMCP("memory-mcp-sse")

@mcp.tool()
def add_memory(content: str, tags: list = None, category: str = None) -> int:
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
def search_memories(query: str, k: int = 5) -> list:
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
def update_memory(memory_id: int, content: str = None, tags: list = None, category: str = None) -> bool:
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

def create_starlette_app(mcp_server: Server, *, debug: bool = False) -> Starlette:
    """Create a Starlette application that can serve the provided mcp server with SSE."""
    sse = SseServerTransport("/messages/")

    async def handle_sse(request: Request) -> None:
        async with sse.connect_sse(
                request.scope,
                request.receive,
                request._send,
        ) as (read_stream, write_stream):
            await mcp_server.run(
                read_stream,
                write_stream,
                mcp_server.create_initialization_options(),
            )

    return Starlette(
        debug=debug,
        routes=[
            Route("/sse", endpoint=handle_sse),
            Mount("/messages/", app=sse.handle_post_message),
        ],
    )

if __name__ == "__main__":
    # Check for required environment variables
    if not os.getenv('DATABASE_URL'):
        logger.error("DATABASE_URL environment variable not set")
        sys.exit(1)

    mcp_server = mcp._mcp_server

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run Memory MCP SSE-based server')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=18080, help='Port to listen on')
    args = parser.parse_args()

    # Create and run Starlette application
    starlette_app = create_starlette_app(mcp_server, debug=True)
    logger.info(f"Starting memory server on {args.host}:{args.port}")
    uvicorn.run(starlette_app, host=args.host, port=args.port)