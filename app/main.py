import asyncio
import logging
import sys
import os

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
# from langgraph.checkpoint.memory import MemorySaver

from app.core.config import settings
import app.core.database as db
from app.core import index_db
from app import bot_handlers
from app.bot_handlers.commands import setup_bot_commands


def setup_handlers(dp: Dispatcher) -> None:
    dp.include_routers(bot_handlers.CommandRouter, bot_handlers.MessageRouter)


async def aiogram_on_startup_polling(dispatcher: Dispatcher, bot: Bot) -> None:
    await bot.delete_webhook(drop_pending_updates=True)
    setup_handlers(dispatcher)
    await setup_bot_commands(bot)
    db.setup_db()
    #await db.populate_db_with_fake_data()
    #index_db.drop_index_db()
    index_db.populate_index_db()
    #index_db.test_index_db()


async def aiogram_on_shutdown_polling(dispatcher: Dispatcher, bot: Bot) -> None:
    #await close_db_connections(dispatcher)
    await bot.session.close()


def main() -> None:
    logging.basicConfig(level=logging.CRITICAL, stream=sys.stdout)

    bot = Bot(
        settings.TELEGRAM_TOKEN,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML)
    )

    dp = Dispatcher()

    if settings.USE_WEBHOOK:
        #TODO: Implement webhook mode
        raise NotImplementedError
    else:
        dp.startup.register(aiogram_on_startup_polling)
        dp.shutdown.register(aiogram_on_shutdown_polling)
        
        asyncio.run(dp.start_polling(bot))


if __name__ == "__main__":
    main()
