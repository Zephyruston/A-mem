from pg_memory_system import PGMemorySystem
import os

# Initialize memory system
memory_system = PGMemorySystem(
    model_name='all-MiniLM-L6-v2',
    llm_backend="openai",
    llm_model="qwen-max-latest",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1/",
    db_url=os.getenv('DATABASE_URL')
)

# Example memories
memories = [
    {
        "content": "I have run the Guangzhou Marathon",
        "tags": ["marathon", "running"],
        "category": "Sports"
    },
    {
        "content": "My favorite food is chicken wings",
        "tags": ["food", "favorite"],
        "category": "Personal"
    },
    {
        "content": "I like R&B music",
        "tags": ["music", "preference"],
        "category": "Personal"
    },
    {
        "content": "I want to see Mount Fuji",
        "tags": ["travel", "Japan"],
        "category": "Personal"
    },
    {
        "content": "I am familiar with Docker Compose and have set up a MySQL master-slave structure for testing",
        "tags": ["Docker", "MySQL"],
        "category": "Technology"
    }
]

# Add memories
print("Adding memories...")
memory_ids = []
for memory in memories:
    memory_id = memory_system.add_memory(
        content=memory["content"],
        tags=memory["tags"],
        category=memory["category"]
    )
    memory_ids.append(memory_id)
    print(f"Added memory {memory_id}: {memory['content']}")

# Search memories
print("\nSearching for memories about technology...")
results = memory_system.search_memories("technology", k=2)
for result in results:
    print(f"\nID: {result['id']}")
    print(f"Content: {result['content']}")
    print(f"Tags: {result['metadata']['tags']}")
    print(f"Category: {result['metadata']['category']}")
    print(f"Similarity: {result['similarity']:.4f}")

# Update a memory
print("\nUpdating a memory...")
memory_id = memory_ids[0]
success = memory_system.update_memory(
    memory_id,
    content="I have completed the Guangzhou Marathon",
    tags=["marathon", "running", "achievement"]
)
if success:
    print(f"Updated memory {memory_id}")
    
    # Get updated memory
    memory = memory_system.get_memory(memory_id)
    print(f"Updated content: {memory['content']}")
    print(f"Updated tags: {memory['metadata']['tags']}")

# Delete a memory
print("\nDeleting a memory...")
memory_id = memory_ids[-1]
success = memory_system.delete_memory(memory_id)
if success:
    print(f"Deleted memory {memory_id}")
    
    # Verify deletion
    memory = memory_system.get_memory(memory_id)
    print(f"Memory exists: {memory is not None}") 