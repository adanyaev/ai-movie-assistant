from aiogram import Bot, Router, html
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import Command
from aiogram.types import Message

from app.core.config import settings
from app.core.database import async_session_factory
from app import crud
from app.agent.llms import LLMFactory
from app.agent.nodes.autonomous_task import RecommendUsersAutonomousTask

router = Router(name="commands-router") 

@router.message(
    Command("start")
)
async def start(message: Message):
    tg_chat_id = message.chat.id
    async with async_session_factory() as session:
        curr_user = await crud.user.get_by_tg_chat_id(session, tg_chat_id)
        if not curr_user:
            curr_user = await crud.user.create(session, {"full_name": message.from_user.full_name, "tg_chat_id": tg_chat_id})


    text = f"Hello, {html.bold(curr_user.full_name)}!"
    await message.answer(text)


@router.message(
    Command("test_autonomous_task")
)
async def test_autonomous_task(message: Message):
    bot = Bot(
        settings.TELEGRAM_TOKEN,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML)
    )
    llm = LLMFactory.get_llm(settings.LLM_NAME)

    async with async_session_factory() as session:
        task = RecommendUsersAutonomousTask(llm, show_logs=True)
        await task.ainvoke(session, bot, message.chat.id)

    await message.answer("Test autonomous task triggered.")
