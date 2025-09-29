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

"""
async def example_multi_agent_prompt() -> None:
    msgs = [
        Msg("system", "你是一个名为 Bob 的有用助手。", "system"),
        Msg("Alice", "嗨！", "user"),
        Msg("Bob", "嗨！很高兴见到大家。", "assistant"),
        Msg("Charlie", "我也是！顺便说一下，我是 Charlie。", "assistant"),
    ]

    formatter = DashScopeMultiAgentFormatter()
    prompt = await formatter.format(msgs)

    print("格式化的提示：")
    print(json.dumps(prompt, indent=4, ensure_ascii=False))

    # 我们在这里打印组合用户消息的内容以便更好地理解：
    print("-------------")
    print("组合消息")
    print(prompt[1]["content"])


asyncio.run(example_multi_agent_prompt())
"""

model = DashScopeChatModel(
    model_name="qwen-max",
    api_key=os.environ["DASHSCOPE_API_KEY"],
)
formatter = DashScopeMultiAgentFormatter()

alice = ReActAgent(
    name="Alice",
    sys_prompt="你是一个名为 Alice 的学生。",
    model=model,
    formatter=formatter,
)

bob = ReActAgent(
    name="Bob",
    sys_prompt="你是一个名为 Bob 的学生。",
    model=model,
    formatter=formatter,
)

charlie = ReActAgent(
    name="Charlie",
    sys_prompt="你是一个名为 Charlie 的学生。",
    model=model,
    formatter=formatter,
)


async def example_msghub() -> None:
    """使用 MsgHub 进行多智能体对话的示例。"""
    async with MsgHub(
        [alice, bob, charlie],
        # 进入 MsgHub 时的公告消息
        announcement=Msg(
            "system",
            "现在大家互相认识一下，简单自我介绍。",
            "system",
        ),
    ):
        await alice()
        await bob()
        await charlie()
async def example_memory() -> None:
    """打印 Alice 的记忆。"""
    print("Alice 的记忆：")
    for msg in await alice.memory.get_memory():
        print(
            f"{msg.name}: {json.dumps(msg.content, indent=4, ensure_ascii=False)}",
        )

asyncio.run(example_msghub())


asyncio.run(example_memory())