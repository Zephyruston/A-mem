# Agentic Memory 🧠

A novel agentic memory system for LLM agents that can dynamically organize memories in an agentic way.

## Introduction 🌟

Large Language Model (LLM) agents have demonstrated remarkable capabilities in handling complex real-world tasks through external tool usage. However, to effectively leverage historical experiences, they require sophisticated memory systems. Traditional memory systems, while providing basic storage and retrieval functionality, often lack advanced memory organization capabilities.

Our project introduces an innovative **Agentic Memory** system that revolutionizes how LLM agents manage and utilize their memories:

<div align="center">
  <img src="Figure/intro-a.jpg" alt="Traditional Memory System" width="600"/>
  <img src="Figure/intro-b.jpg" alt="Our Proposed Agentic Memory" width="600"/>
  <br>
  <em>Comparison between traditional memory system (top) and our proposed agentic memory (bottom). Our system enables dynamic memory operations and flexible agent-memory interactions.</em>
</div>

> **Note:** This repository provides a memory system to facilitate agent construction. If you want to reproduce the results presented in our paper, please refer to: [https://github.com/WujiangXu/AgenticMemory](https://github.com/WujiangXu/AgenticMemory)

For more details, please refer to our paper: [A-MEM: Agentic Memory for LLM Agents](https://arxiv.org/pdf/2502.12110)

## Key Features ✨

- 🔄 Dynamic memory organization based on Zettelkasten principles
- 🔍 Intelligent indexing and linking of memories via ChromaDB
- 📝 Comprehensive note generation with structured attributes
- 🌐 Interconnected knowledge networks
- 🧬 Continuous memory evolution and refinement
- 🤖 Agent-driven decision making for adaptive memory management

## Framework 🏗️

<div align="center">
  <img src="Figure/framework.jpg" alt="Agentic Memory Framework" width="800"/>
  <br>
  <em>The framework of our Agentic Memory system showing the dynamic interaction between LLM agents and memory components.</em>
</div>

## How It Works 🛠️

When a new memory is added to the system:

1. Generates comprehensive notes with structured attributes
2. Creates contextual descriptions and tags
3. Analyzes historical memories for relevant connections
4. Establishes meaningful links based on similarities
5. Enables dynamic memory evolution and updates

## Results 📊

Empirical experiments conducted on six foundation models demonstrate superior performance compared to existing SOTA baselines.

## Getting Started 🚀

1. Clone the repository:

```bash
git clone https://github.com/agiresearch/A-mem.git
cd AgenticMemory
```

2. Install dependencies:
   Option 1: Using venv (Python virtual environment)

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

Option 2: Using Conda

```bash
# Create and activate conda environment
conda create -n myenv python=3.9
conda activate myenv

# Install dependencies
pip install -r requirements.txt
```

3. Usage Examples 💡

set your openai api key

```bash
export OPENAI_API_KEY=your_api_key
```

and then run example.py(optional: you can put your own memory_list in example.py)

```bash
python example.py
```

Here's how to use the Agentic Memory system for basic operations:

```python
from memory_system import AgenticMemorySystem

# Initialize the memory system 🚀
memory_system = AgenticMemorySystem(
    model_name='all-MiniLM-L6-v2',  # Embedding model for ChromaDB
    llm_backend="openai",           # LLM backend (openai/ollama)
    llm_model="gpt-4o-mini"         # LLM model name
)

# Add Memories ➕
# Simple addition
memory_id = memory_system.add_note("Deep learning neural networks")

# Addition with metadata
memory_id = memory_system.add_note(
    content="Machine learning project notes",
    tags=["ml", "project"],
    category="Research",
    timestamp="202503021500"  # YYYYMMDDHHmm format
)

# Read (Retrieve) Memories 📖
# Get memory by ID
memory = memory_system.read(memory_id)
print(f"Content: {memory.content}")
print(f"Tags: {memory.tags}")
print(f"Context: {memory.context}")
print(f"Keywords: {memory.keywords}")

# Search memories
results = memory_system.search_agentic("neural networks", k=5)
for result in results:
    print(f"ID: {result['id']}")
    print(f"Content: {result['content']}")
    print(f"Tags: {result['tags']}")
    print("---")

# Update Memories 🔄
memory_system.update(memory_id, content="Updated content about deep learning")

# Delete Memories ❌
memory_system.delete(memory_id)

# Memory Evolution 🧬
# The system automatically evolves memories by:
# 1. Finding semantic relationships using ChromaDB
# 2. Updating metadata and context
# 3. Creating connections between related memories
# This happens automatically when adding or updating memories!
```

### Advanced Features 🌟

1. **ChromaDB Vector Storage** 📦

   - Efficient vector embedding storage and retrieval
   - Fast semantic similarity search
   - Automatic metadata handling
   - Persistent memory storage

2. **Memory Evolution** 🧬

   - Automatically analyzes content relationships
   - Updates tags and context based on related memories
   - Creates semantic connections between memories

3. **Flexible Metadata** 📋

   - Custom tags and categories
   - Automatic keyword extraction
   - Context generation
   - Timestamp tracking

4. **Multiple LLM Backends** 🤖
   - OpenAI (GPT-4, GPT-3.5)
   - Ollama (for local deployment)

### Best Practices 💪

1. **Memory Creation** ✨:

   - Provide clear, specific content
   - Add relevant tags for better organization
   - Let the system handle context and keyword generation

2. **Memory Retrieval** 🔍:

   - Use specific search queries
   - Adjust 'k' parameter based on needed results
   - Consider both exact and semantic matches

3. **Memory Evolution** 🧬:

   - Allow automatic evolution to organize memories
   - Review generated connections periodically
   - Use consistent tagging conventions

4. **Error Handling** ⚠️:
   - Always check return values
   - Handle potential KeyError for non-existent memories
   - Use try-except blocks for LLM operations

## Citation 📚

If you use this code in your research, please cite our work:

```bibtex
@article{xu2025mem,
  title={A-mem: Agentic memory for llm agents},
  author={Xu, Wujiang and Liang, Zujie and Mei, Kai and Gao, Hang and Tan, Juntao and Zhang, Yongfeng},
  journal={arXiv preprint arXiv:2502.12110},
  year={2025}
}
```

## License 📄

This project is licensed under the MIT License. See LICENSE for details.

### MCP Tool Usage 🛠️

The memory system can be used as an MCP tool, allowing natural language interaction with the memory system.

#### Prerequisites

1. Environment variables:

```bash
# PostgreSQL connection
export DATABASE_URL="postgresql://user:password@localhost:5432/dbname"

# LLM API configuration
export API_KEY="your-api-key"
export MODEL_NAME="qwen-max-latest"
export BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1/"
```

#### Running the System

1. Start the memory server:

```bash
python memory_server.py
```

2. Start the memory client:

```bash
python memory_client.py
```

#### Example Interactions

Once the client is running, you can interact with the memory system using natural language:

1. Adding memories:

   - "我想记住我喜欢吃披萨"
   - "添加一条记忆：我最近在学习 Python 编程"

2. Searching memories:

   - "搜索关于食物的记忆"
   - "查找所有与编程相关的记忆"

3. Updating memories:

   - "更新记忆 1 的内容为：我最喜欢吃意大利披萨"
   - "修改记忆 2 的标签为：编程、Python、学习"

4. Deleting memories:
   - "删除记忆 1"
   - "移除 ID 为 3 的记忆"

The system will automatically convert your natural language requests into appropriate tool calls and display the results.
