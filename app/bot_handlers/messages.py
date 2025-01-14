from aiogram import Router
from aiogram.types import Message

from app.core.config import settings
from app.agent import agent_instance, build_state
from app.core.database import async_session_factory
from app import crud


router = Router(name="messages-router")

@router.message()
async def general_handler(message: Message) -> None:
    text = message.text
    try:

        status_message = await message.answer("Обрабатываю Ваш запрос, подождите... 🔎")

        async with async_session_factory() as session:
            # Save user's message
            curr_user = await crud.user.get_by_tg_chat_id(session, message.chat.id)
            if not curr_user:
                curr_user = await crud.user.create(session, {"full_name": message.from_user.full_name, "tg_chat_id": message.chat.id})
            await crud.message.create(
                session, 
                {
                    "user_id": curr_user.id,
                    "content": text,
                    "message_type": "human"
                }
            )

            # Refresh user
            curr_user = await crud.user.get_by_tg_chat_id(session, message.chat.id)
        
            messages = curr_user.messages[-settings.USER_HISTORY_LIMIT:]
            state = build_state([(msg.message_type.value, msg.content) for msg in messages], curr_user.id)
            new_state = agent_instance.invoke(state)

            answ_type, answ_text = new_state.history[-1].type, new_state.history[-1].content
            
            # Save bot's response
            await crud.message.create(
                session,
                {
                    "user_id": curr_user.id,
                    "content": answ_text,
                    "message_type": answ_type
                }
            )
            
            # Edit the status message with the final response
            await status_message.edit_text(answ_text)

    except Exception as e:
        print(e)
        await message.answer("Internal error")
