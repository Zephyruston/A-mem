import asyncio
import json
import os
import sys
from typing import List, Dict, Any
from mcp import ClientSession
from mcp.client.sse import sse_client
from openai import AsyncOpenAI

class MemoryClient:
    def __init__(self, model_name: str, base_url: str, api_key: str, server_url: str):
        """
        初始化记忆系统客户端。

        Args:
            model_name: 使用的模型名称
            base_url: OpenAI 接口的基础地址
            api_key: OpenAI API 密钥
            server_url: MCP 服务器地址
        """
        self.model_name = model_name
        self.server_url = server_url
        self.sessions = {}
        self.tool_mapping = {}

        # 初始化 OpenAI 异步客户端
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)

    async def initialize_session(self):
        """初始化与 MCP 服务器的连接并获取可用工具列表。"""
        # 创建 SSE 客户端并进入上下文
        streams_context = sse_client(url=self.server_url)
        streams = await streams_context.__aenter__()
        session_context = ClientSession(*streams)
        session = await session_context.__aenter__()
        await session.initialize()

        # 存储会话及其上下文
        self.sessions["memory_server"] = (session, session_context, streams_context)

        # 获取工具列表并建立映射
        response = await session.list_tools()
        for tool in response.tools:
            self.tool_mapping[tool.name] = (session, tool.name)
        print(f"已连接到 {self.server_url}，可用工具：{[tool.name for tool in response.tools]}")

    async def cleanup(self):
        """清理所有会话和连接资源。"""
        for server_id, (session, session_context, streams_context) in self.sessions.items():
            await session_context.__aexit__(None, None, None)
            await streams_context.__aexit__(None, None, None)
        print("所有会话已清理。")

    async def process_query(self, query: str) -> str:
        """
        处理用户的自然语言查询，通过工具调用完成任务并返回结果。

        Args:
            query: 用户输入的查询字符串

        Returns:
            str: 处理后的回复文本
        """
        messages = [{"role": "user", "content": query}]

        # 收集所有可用工具
        available_tools = []
        for server_id, (session, _, _) in self.sessions.items():
            response = await session.list_tools()
            for tool in response.tools:
                available_tools.append({
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.inputSchema,
                    },
                })

        # 向模型发送初始请求
        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            tools=available_tools,
        )

        final_text = []
        message = response.choices[0].message
        final_text.append(message.content or "")

        # 处理工具调用
        while message.tool_calls:
            for tool_call in message.tool_calls:
                tool_name = tool_call.function.name
                if tool_name in self.tool_mapping:
                    session, original_tool_name = self.tool_mapping[tool_name]
                    tool_args = json.loads(tool_call.function.arguments)
                    try:
                        result = await session.call_tool(original_tool_name, tool_args)
                    except Exception as e:
                        result = {"content": f"调用工具 {original_tool_name} 出错：{str(e)}"}
                        print(result["content"])
                    final_text.append(f"[调用工具 {tool_name} 参数: {tool_args}]")
                    final_text.append(f"工具结果: {result.content}")
                    messages.extend([
                        {
                            "role": "assistant",
                            "tool_calls": [{
                                "id": tool_call.id,
                                "type": "function",
                                "function": {"name": tool_name, "arguments": json.dumps(tool_args)},
                            }],
                        },
                        {"role": "tool", "tool_call_id": tool_call.id, "content": str(result.content)},
                    ])
                else:
                    print(f"工具 {tool_name} 未找到")
                    final_text.append(f"工具 {tool_name} 未找到")

            # 获取工具调用后的后续回复
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                tools=available_tools,
            )
            message = response.choices[0].message
            if message.content:
                final_text.append(message.content)

        return "\n".join(final_text)

    async def chat_loop(self):
        """启动命令行交互式对话循环。"""
        print("\n记忆系统客户端已启动，输入你的问题，输入 'quit' 退出。")
        print("示例问题：")
        print("1. 添加一条记忆：'我想记住我喜欢吃披萨'")
        print("2. 搜索记忆：'搜索关于食物的记忆'")
        print("3. 更新记忆：'更新记忆1的内容为：我最喜欢吃意大利披萨'")
        print("4. 删除记忆：'删除记忆1'")
        
        while True:
            try:
                query = input("\n问题: ").strip()
                if query.lower() == "quit":
                    break
                response = await self.process_query(query)
                print("\n" + response)
            except Exception as e:
                print(f"\n发生错误: {str(e)}")

async def main():
    """程序入口，设置配置并启动记忆系统客户端。"""
    # 从环境变量获取配置
    model_name = os.getenv("MODEL_NAME", "qwen-max-latest")
    base_url = os.getenv("BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1/")
    api_key = os.getenv("API_KEY")
    if not api_key:
        print("未设置 API_KEY 环境变量。")
        sys.exit(1)

    # MCP 服务器地址
    server_url = "http://localhost:18080/sse"

    # 创建并运行客户端
    client = MemoryClient(
        model_name=model_name,
        base_url=base_url,
        api_key=api_key,
        server_url=server_url
    )
    try:
        await client.initialize_session()
        await client.chat_loop()
    finally:
        await client.cleanup()

if __name__ == "__main__":
    asyncio.run(main()) 