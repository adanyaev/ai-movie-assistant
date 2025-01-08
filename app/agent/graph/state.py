from typing import List

from langchain_core.pydantic_v1 import BaseModel
from langchain_core.messages import BaseMessage


class AgentState(BaseModel):
    history: List[BaseMessage]
    user_id: str
