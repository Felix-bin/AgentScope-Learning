import asyncio
import json
import os
from typing import Literal
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from agentscope.agent import ReActAgent
from agentscope.formatter import DashScopeChatFormatter
from agentscope.memory import InMemoryMemory
from agentscope.message import Msg
from agentscope.model import DashScopeChatModel
from agentscope.tool import Toolkit, ToolResponse

load_dotenv()

router = ReActAgent(
    name="Router",
    sys_prompt="你是一个路由智能体。你的目标是将用户查询路由到正确的后续任务，注意你不需要回答用户的问题。",
    model=DashScopeChatModel(
        model_name="qwen-max",
        api_key=os.environ["DASHSCOPE_API_KEY"],
        stream=False,
    ),
    formatter=DashScopeChatFormatter(),
)


# 使用结构化输出指定路由任务
class RoutingChoice(BaseModel):
    your_choice: Literal[
        "Content Generation",
        "Programming",
        "Information Retrieval",
        None,
    ] = Field(
        description="选择正确的后续任务，如果任务太简单或没有合适的任务，则选择 ``None``",
    )
    task_description: str | None = Field(
        description="任务描述",
        default=None,
    )

async def generate_python(demand: str) -> ToolResponse:
    """根据需求生成 Python 代码。

    Args:
        demand (``str``):
            对 Python 代码的需求。
    """
    # 示例需求智能体
    python_agent = ReActAgent(
        name="PythonAgent",
        sys_prompt="你是一个 Python 专家，你的目标是根据需求生成 Python 代码。",
        model=DashScopeChatModel(
            model_name="qwen-max",
            api_key=os.environ["DASHSCOPE_API_KEY"],
            stream=False,
        ),
        memory=InMemoryMemory(),
        formatter=DashScopeChatFormatter(),
        toolkit=Toolkit(),
    )
    msg_res = await python_agent(Msg("user", demand, "user"))

    return ToolResponse(
        content=msg_res.get_content_blocks("text"),
    )


# 为演示目的模拟一些其他工具函数
async def generate_poem(demand: str) -> ToolResponse:
    """根据需求生成诗歌。

    Args:
        demand (``str``):
            对诗歌的需求。
    """
    pass


async def web_search(query: str) -> ToolResponse:
    """在网络上搜索查询。

    Args:
        query (``str``):
            要搜索的查询。
    """
    pass


toolkit = Toolkit()
toolkit.register_tool_function(generate_python)
toolkit.register_tool_function(generate_poem)
toolkit.register_tool_function(web_search)

# 使用工具模块初始化路由智能体
router_implicit = ReActAgent(
    name="Router",
    sys_prompt="你是一个路由智能体。你的目标是将用户查询路由到正确的后续任务。",
    model=DashScopeChatModel(
        model_name="qwen-max",
        api_key=os.environ["DASHSCOPE_API_KEY"],
        stream=False,
    ),
    formatter=DashScopeChatFormatter(),
    toolkit=toolkit,
    memory=InMemoryMemory(),
)


async def example_router_implicit() -> None:
    """使用工具调用进行隐式路由的示例。"""
    msg_user = Msg(
        "user",
        "帮我在 Python 中生成一个快速排序函数",
        "user",
    )

    # 路由查询
    await router_implicit(msg_user)


asyncio.run(example_router_implicit())
