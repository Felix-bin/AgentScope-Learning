from agentscope.agent import ReActAgent, AgentBase
from agentscope.formatter import DashScopeChatFormatter
from agentscope.memory import InMemoryMemory
from agentscope.message import Msg
from agentscope.model import DashScopeChatModel
from agentscope.tool import Toolkit, execute_python_code
from dotenv import load_dotenv
import asyncio
import os

#加载 .env里面的所有环境变量
load_dotenv()


async def creating_react_agent() -> None:
    """创建一个 ReAct 智能体并运行一个简单任务。"""
    # 准备工具
    toolkit = Toolkit()
    toolkit.register_tool_function(execute_python_code)

    jarvis = ReActAgent(
        name="Jarvis",
        sys_prompt="你是一个名为 Jarvis 的助手",
        model=DashScopeChatModel(
            model_name="qwen-max",
            api_key=os.environ["DASHSCOPE_API_KEY"],
            stream=True,
            enable_thinking=True,
        ),
        formatter=DashScopeChatFormatter(),
        toolkit=toolkit,
        memory=InMemoryMemory(),
    )

    msg = Msg(
        name="user",
        content="你好！Jarvis，用 Python 运行 Hello World。",
        role="user",
    )

    await jarvis(msg)


asyncio.run(creating_react_agent())