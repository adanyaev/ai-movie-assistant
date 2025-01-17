from typing import List

from langchain_core.messages import BaseMessage
from pydantic import BaseModel

from app.schemas.user import UserPreferenceBase

class AgentState(BaseModel):
    history: List[BaseMessage]
    user_id: int
    user_preferences: List[UserPreferenceBase]
