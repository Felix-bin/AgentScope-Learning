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

class MyAgent(AgentBase):
    """自定义智能体类"""

    def __init__(self) -> None:
        """初始化智能体"""
        super().__init__()

        self.name = "Friday"
        self.sys_prompt = "你是一个名为 Friday 的助手。"
        self.model = DashScopeChatModel(
            model_name="qwen-max",
            api_key=os.environ["DASHSCOPE_API_KEY"],
            stream=False,
        )
        self.formatter = DashScopeChatFormatter()
        self.memory = InMemoryMemory()

    async def reply(self, msg: Msg | list[Msg] | None) -> Msg:
        """直接调用大模型，产生回复消息。"""
        await self.memory.add(msg)

        # 准备提示
        prompt = await self.formatter.format(
            [
                Msg("system", self.sys_prompt, "system"),
                *await self.memory.get_memory(),
            ],
        )

        # 调用模型
        response = await self.model(prompt)

        msg = Msg(
            name=self.name,
            content=response.content,
            role="assistant",
        )

        # 在记忆中记录响应
        await self.memory.add(msg)

        # 打印消息
        await self.print(msg)
        return msg

    async def observe(self, msg: Msg | list[Msg] | None) -> None:
        """观察消息。"""
        # 将消息存储在记忆中
        await self.memory.add(msg)

    async def handle_interrupt(self) -> Msg:
        """处理中断。"""
        # 以固定响应为例
        return Msg(
            name=self.name,
            content="我注意到您打断了我的回复，我能为你做些什么？",
            role="assistant",
        )


async def run_custom_agent() -> None:
    """运行自定义智能体。"""
    agent = MyAgent()
    msg = Msg(
        name="user",
        content="你是谁？",
        role="user",
    )
    await agent(msg)


asyncio.run(run_custom_agent())