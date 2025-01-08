import asyncio
import logging
import sys
import os

from aiogram import Bot, Dispatcher, html
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart
from aiogram.types import Message

from psycopg import AsyncConnection
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
#from langgraph.checkpoint.memory import MemorySaver

from app.core.config import settings
import app.core.database as db
import app.agent.agent as agent


TOKEN = os.environ.get("TELEGRAM_TOKEN")

# All handlers should be attached to the Router (or Dispatcher)

dp = Dispatcher()


@dp.message(CommandStart())
async def command_start_handler(message: Message) -> None:
    """
    This handler receives messages with `/start` command
    """
    await message.answer(f"Hello, {html.bold(message.from_user.full_name)}!")


@dp.message()
async def echo_handler(message: Message) -> None:
    """
    Handler will forward receive a message back to the sender

    By default, message handler will handle all message types (like a text, photo, sticker etc.)
    """
    
    async with await AsyncConnection.connect(settings.DB_URI, **agent.connection_kwargs) as conn:
        checkpointer = AsyncPostgresSaver(conn)
        #await checkpointer.setup()
        #checkpointer = MemorySaver()
        try:
            text = message.text
            app = agent.workflow.compile(checkpointer=checkpointer)
            final_state = await app.ainvoke(
                {"messages": [("human", text)]},
                config={"configurable": {"thread_id": str(message.chat.id)}}
            )
            answ_text = final_state["messages"][-1].content
            await message.answer(answ_text)
        except Exception as e:
            print(e)
            await message.answer("Internal error")


async def main() -> None:
    # Initialize Bot instance with default bot properties which will be passed to all API calls
    bot = Bot(token=TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
    db.setup_db()
    print(settings.DB_URI)
    await db.populate_db_with_fake_data()
    # And the run events dispatching
    await dp.start_polling(bot)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
    asyncio.run(main())
