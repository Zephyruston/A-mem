from memory_system import AgenticMemorySystem

# 输入的记忆列表
memory_list = [
    {
        "content": "I have run the Guangzhou Marathon",
        "tags": ["marathon", "running"],
        "category": "Sports",
        "timestamp": "202506150900"
    },
    {
        "content": "My favorite food is chicken wings",
        "tags": ["food", "favorite"],
        "category": "Personal",
        "timestamp": "202506150904"
    },
    {
        "content": "I like R&B music",
        "tags": ["music", "preference"],
        "category": "Personal",
        "timestamp": "202506150905"
    },
    {
        "content": "I want to see Mount Fuji",
        "tags": ["travel", "Japan"],
        "category": "Personal",
        "timestamp": "202506150908"
    },
    {
        "content": "I am familiar with Docker Compose and have set up a MySQL master-slave structure for testing",
        "tags": ["Docker", "MySQL"],
        "category": "Technology",
        "timestamp": "202506150928"
    }
]

# 初始化内存系统
memory_system = AgenticMemorySystem(
    model_name='all-MiniLM-L6-v2',  # ChromaDB 的嵌入模型
    llm_backend="openai",           # LLM 后端 (openai/ollama)
    llm_model="qwen-max-latest",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1/"
)

# 批量添加记忆
added_memory_ids = []
for memory in memory_list:
    memory_id = memory_system.add_note(**memory)
    added_memory_ids.append(memory_id)

# 交互式搜索界面
while True:
    query = input("请输入搜索关键词（输入 'q' 退出）: ")
    if query.lower() == 'q':
        break
    k = int(input("请输入要返回的结果数量: "))
    results = memory_system.search_agentic(query, k=k)
    if results:
        for result in results:
            print(f"ID: {result['id']}")
            print(f"内容: {result['content']}")
            print(f"标签: {result['tags']}")
            print(f"上下文: {result['context']}")
            print(f"关键词: {result['keywords']}")
            print("---")
    else:
        print("未找到相关记忆。")

