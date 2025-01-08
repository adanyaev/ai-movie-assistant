from datetime import datetime
from pydantic import BaseModel, ConfigDict

from app.models.user import PreferenceItem, PreferenceType
from .message import Message

class UserPreferenceBase(BaseModel):
    preference_item: PreferenceItem
    preference_type: PreferenceType
    item_name: str

class UserPreference(UserPreferenceBase):
    id: int
    user_id: int
    #user: "User"

    model_config = ConfigDict(from_attributes=True)

class User(BaseModel):
    id: int
    full_name: str
    is_active: bool
    is_superuser: bool
    created_at: datetime
    messages: list[Message]
    preferences: list[UserPreference]

    model_config = ConfigDict(from_attributes=True)

class UserCreate(BaseModel):
    full_name: str
    is_active: bool = True
    is_superuser: bool = False
    favorite_films: list[str] = []
    favorite_genres: list[str] = []

class UserUpdate(BaseModel):
    full_name: str | None = None
    is_active: bool | None = None
    favorite_films: list[str] | None = None
    favorite_genres: list[str] | None = None