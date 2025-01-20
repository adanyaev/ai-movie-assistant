from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker

from app import crud
from app.core.config import settings
from app.models.base import Base
from app.models.user import PreferenceItem, PreferenceType
from app.models.message import MessageType


async_engine = create_async_engine(
    settings.ASYNC_DB_URI,
    echo=settings.VERBOSE_DB,
)

engine = create_engine(
        settings.DB_URI,
        echo=settings.VERBOSE_DB,
    )

async_session_factory = async_sessionmaker(async_engine)

def init_db() -> None:

    engine = create_engine(
        settings.DB_URI,
        echo=settings.VERBOSE_DB,
    )
    Base.metadata.create_all(engine)


def drop_db() -> None:

    engine = create_engine(
        settings.DB_URI,
        echo=settings.VERBOSE_DB,
    )
    Base.metadata.drop_all(engine)


def setup_db() -> None:
    if settings.DROP_DB:
        drop_db()
        print("Database drop complete")
    init_db()
    print("Database setup complete")


async def populate_db_with_fake_data() -> None:
    import random
    async with async_session_factory() as session:
        # Create users
        users = []
        for i in range(5):
            user_data = {
                "full_name": f"User {i+1}",
                "tg_chat_id": random.randint(1000, 999999),
                "is_active": False
            }
            user = await crud.user.create(session, user_data)
            users.append(user)

            # Add preferences for each user
            for ptype in PreferenceType:
                item = PreferenceItem.MOVIE if i % 2 == 0 else PreferenceItem.GENRE
                await crud.user.create_preference(
                    session, 
                    user.id,
                    -1,
                    f"test pref {item.value} type",
                    item,
                    ptype
                )

            # Add messages for each user
            for j in range(3):
                message_data = {
                    "user_id": user.id,
                    "content": f"Message {j+1} for User {i+1}",
                    "message_type": MessageType.HUMAN if j % 2 == 0 else MessageType.AI
                }
                await crud.message.create(session, message_data)

    print("Database populated with fake data")
