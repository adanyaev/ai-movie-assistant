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
from app import bot_handlers


def setup_handlers(dp: Dispatcher) -> None:
    dp.include_routers(bot_handlers.CommandRouter, bot_handlers.MessageRouter)


async def aiogram_on_startup_polling(dispatcher: Dispatcher, bot: Bot) -> None:
    await bot.delete_webhook(drop_pending_updates=True)
    setup_handlers(dispatcher)
    db.setup_db()
    await db.populate_db_with_fake_data()


async def aiogram_on_shutdown_polling(dispatcher: Dispatcher, bot: Bot) -> None:
    #await close_db_connections(dispatcher)
    await bot.session.close()


def main() -> None:
    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)

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
