from aiogram import Router, html
from aiogram.filters import Command
from aiogram.types import Message

from app.core.database import async_session_factory
from app import crud

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
