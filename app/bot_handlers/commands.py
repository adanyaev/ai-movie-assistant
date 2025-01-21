from aiogram import Bot, Router, html
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import Command
from aiogram.types import Message, BotCommand

from app.core.config import settings
from app.core.database import async_session_factory
from app import crud
from app.agent.llms import LLMFactory
from app.agent.nodes.autonomous_task import RecommendUsersAutonomousTask

router = Router(name="commands-router") 


async def setup_bot_commands(bot: Bot):
    commands = [
        BotCommand(command="start", description="Приветствие"),
        BotCommand(command="test_autonomous_task", description="Запуск тестовой автономной задачи"),
        BotCommand(command="help", description="Показать доступные команды")
    ]
    await bot.set_my_commands(commands)


HELLO_MESSAGE = """
👋 Привет, {user_name}! Я — твой дружелюбный помощник по всему, что связано с миром кино и сериалов! 🎥✨

Моя задача — помогать тебе находить информацию о фильмах, сериалах, актерах, режиссерах и других аспектах киноиндустрии. Вот что я умею:

🎬 Рассказывать о фильмах и сериалах
Могу найти описание, рейтинг, жанр или помочь составить подборку по твоим предпочтениям.

🌟 Информация о знаменитостях
Расскажу о жизни и карьере актеров, режиссеров и сценаристов.

🧩 Угадать фильм по описанию сюжета
Если ты помнишь только общие детали, я помогу найти нужный фильм или сериал.

💬 Отзывы зрителей
Могу подытожить мнения о фильмах, чтобы ты узнал, что думают другие.

❤️ Запоминать твои предпочтения
Помогу настроить рекомендации, запомнив, что тебе нравится или не нравится.

Просто задай вопрос, и я постараюсь дать развернутый и полезный ответ! 😊

Например:

«Расскажи о фильме "Интерстеллар".»
«Какие сериалы есть в жанре фантастика?»
«Кто такой Кристофер Нолан?»
«Как называется фильм про советского ученого изобретателя, создавшего машину времени и отправившегося в прошлое?»
Давай начнем! Что тебя интересует? 🎞️
"""


@router.message(
    Command("start")
)
async def start(message: Message):
    tg_chat_id = message.chat.id
    async with async_session_factory() as session:
        curr_user = await crud.user.get_by_tg_chat_id(session, tg_chat_id)
        if not curr_user:
            curr_user = await crud.user.create(session, {"full_name": message.from_user.full_name, "tg_chat_id": tg_chat_id})

    text = HELLO_MESSAGE.format(user_name=html.bold(curr_user.full_name))
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


@router.message(Command("help"))
async def help_command(message: Message):
    text = (
        "Список команд:\n"
        "/start — приветствие\n"
        "/test_autonomous_task — запуск тестовой автономной задачи\n"
        "/help — показать это сообщение"
    )
    await message.answer(text)
