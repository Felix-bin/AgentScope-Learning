import asyncio
import json
import os

from agentscope.agent import ReActAgent, UserAgent
from agentscope.memory import InMemoryMemory
from agentscope.formatter import (
    DashScopeChatFormatter,
    DashScopeMultiAgentFormatter,
)
from agentscope.model import DashScopeChatModel
from agentscope.message import Msg
from agentscope.pipeline import MsgHub
from agentscope.tool import Toolkit

from dotenv import load_dotenv

load_dotenv()

friday = ReActAgent(
    name="Friday",
    sys_prompt="你是一个名为 Friday 的有用助手",
    model=DashScopeChatModel(
        model_name="qwen-max",
        api_key=os.environ["DASHSCOPE_API_KEY"],
    ),
    formatter=DashScopeChatFormatter(),  # 用于 user-assistant 对话的格式化器
    memory=InMemoryMemory(),
    toolkit=Toolkit(),
)

# 创建用户智能体
user = UserAgent(name="User")

async def run_conversation() -> None:
    """运行 Friday 和用户之间的简单对话。"""
    msg = None
    while True:
        msg = await friday(msg)
        msg = await user(msg)
        if msg.get_text_content() == "exit":
            break

asyncio.run(run_conversation())