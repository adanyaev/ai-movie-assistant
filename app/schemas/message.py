from datetime import datetime
from pydantic import BaseModel, ConfigDict

from app.models.message import MessageType

class MessageBase(BaseModel):
    content: str
    message_type: MessageType

class MessageCreate(MessageBase):
    user_id: int

class Message(MessageBase):
    id: int
    user_id: int
    #user: "User"
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
