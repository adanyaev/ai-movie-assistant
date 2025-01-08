from typing import Optional, Sequence, Type

from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.base import Base

class CRUDBase:
    def __init__(self, model: Type[Base], schema: Type[BaseModel]):
        self.model = model
        self.schema = schema

    async def create(self, db: AsyncSession, obj_in: dict) -> BaseModel: # Override this method in the child classes using the schema
        db_obj = self.model(**obj_in)
        db.add(db_obj)
        await db.commit()
        await db.refresh(db_obj)
        return self.schema.model_validate(db_obj)

    async def get(self, db: AsyncSession, id: int) -> Optional[BaseModel]:
        query = select(self.model).where(self.model.id == id)
        result = await db.execute(query)
        scalar = result.scalar_one_or_none()
        if scalar is None:
            return None
        return self.schema.model_validate(scalar)

    async def get_all(self, db: AsyncSession) -> Sequence[BaseModel]:
        query = select(self.model)
        result = await db.execute(query)
        return [self.schema.model_validate(i) for i in result.scalars().all()]

    async def update(self, db: AsyncSession, id: int, obj_in: dict) -> BaseModel:
        db_obj = await self.get(db, id)
        for field, value in obj_in.items():
            setattr(db_obj, field, value)
        await db.commit()
        await db.refresh(db_obj)
        return self.schema.model_validate(db_obj)

    async def delete(self, db: AsyncSession, id: int) -> Optional[BaseModel]:
        obj = await self.get(db, id)
        if obj:
            await db.delete(obj)
            await db.commit()
            return self.schema.model_validate(obj)
        return None
