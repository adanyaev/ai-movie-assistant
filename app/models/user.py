from enum import StrEnum

from sqlalchemy import ForeignKey, Index
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base, created_at

class PreferenceItem(StrEnum):
    MOVIE = "movie"
    GENRE = "genre"
    DIRECTOR = "director"
    ACTOR = "actor"


class PreferenceType(StrEnum):
    LIKE = "like"
    DISLIKE = "dislike"


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    tg_chat_id: Mapped[int] = mapped_column(nullable=False, unique=True)
    full_name: Mapped[str] = mapped_column(nullable=False)
    is_active: Mapped[bool] = mapped_column(nullable=False, default=True)
    is_superuser: Mapped[bool] = mapped_column(nullable=False, default=False)

    created_at: Mapped[created_at]

    messages: Mapped[list["Message"]] = relationship(
        back_populates="user", order_by="Message.created_at", lazy="selectin"
    )
    preferences: Mapped[list["UserPreference"]] = relationship(back_populates="user", lazy="selectin")

    # __table_args__ = (
    #     Index('ix_users_tg_chat_id', 'tg_chat_id'),
    # )


class UserPreference(Base):
    __tablename__ = "user_preferences"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    kp_id: Mapped[int] = mapped_column(nullable=False)
    user: Mapped["User"] = relationship(back_populates="preferences", lazy="joined")

    item_name: Mapped[str] = mapped_column(nullable=False)

    preference_item: Mapped[PreferenceItem] = mapped_column(nullable=False)
    preference_type: Mapped[PreferenceType] = mapped_column(nullable=False)
