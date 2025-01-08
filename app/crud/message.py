from typing import Optional, Sequence, Type

from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.base import Base
from app.models.message import Message, MessageType
from .base import CRUDBase


class CRUDMessage(CRUDBase):
    async def get_user_messages(
        self, db: AsyncSession, user_id: int, message_type: Optional[MessageType] = None
    ) -> Sequence[BaseModel]:
        query = select(self.model).where(self.model.user_id == user_id)
        if message_type:
            query = query.where(self.model.message_type == message_type)
        result = await db.execute(query)
        return [self.schema.model_validate(i) for i in result.scalars().all()]
