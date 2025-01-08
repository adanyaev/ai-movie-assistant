from typing import Optional, Sequence, Type

from pydantic import BaseModel
from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.base import Base
from app.models.user import User, UserPreference, PreferenceItem, PreferenceType
from app.schemas.user import User as UserSchema, UserPreference as UserPreferenceSchema
from .base import CRUDBase


class CRUDUser(CRUDBase):
    async def get_by_fullname(self, db: AsyncSession, full_name: str) -> Optional[UserSchema]:
        query = select(self.model).where(self.model.full_name == full_name)
        result = await db.execute(query)
        scalar = result.scalar_one_or_none()
        if scalar is None:
            return None
        return self.schema.model_validate(scalar)

    async def get_preferences_by_user_id(self, db: AsyncSession, user_id: int) -> list[UserPreferenceSchema]:
        query = select(UserPreference).where(UserPreference.user_id == user_id)
        result = await db.execute(query)
        return [UserPreferenceSchema.model_validate(i) for i in result.scalars().all()]

    async def create_preference(
        self, db: AsyncSession, user_id: int, item_name: str, item: PreferenceItem, ptype: PreferenceType
    ) -> UserPreferenceSchema:
        preference = UserPreference(user_id=user_id, item_name=item_name, preference_item=item, preference_type=ptype)
        db.add(preference)
        await db.commit()
        await db.refresh(preference)
        return UserPreferenceSchema.model_validate(preference)

    async def remove_preference(self, db: AsyncSession, preference_id: int) -> None:
        await db.execute(delete(UserPreference).where(UserPreference.id == preference_id))
        await db.commit()

    # async def update_preference(
    #     self, db: AsyncSession, preference_id: int, item: PreferenceItem, ptype: PreferenceType
    # ) -> Optional[UserPreferenceSchema]:
    #     query = select(UserPreference).where(UserPreference.id == preference_id)
    #     result = await db.execute(query)
    #     pref = result.scalar_one_or_none()
    #     if not pref:
    #         return None
    #     pref.preference_item = item
    #     pref.preference_type = ptype
    #     db.add(pref)
    #     await db.commit()
    #     await db.refresh(pref)
    #     return UserPreferenceSchema.model_validate(pref)
