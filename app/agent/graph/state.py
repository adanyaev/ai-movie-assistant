from typing import List

from langchain_core.messages import BaseMessage
from pydantic import BaseModel


class AgentState(BaseModel):
    history: List[BaseMessage]
    user_id: int
