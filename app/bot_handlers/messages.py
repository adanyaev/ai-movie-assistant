from aiogram import Router
from aiogram.types import Message
from psycopg import AsyncConnection

from app.core.config import settings
from app.agent import agent_instance, build_state
from app.core.database import async_session_factory
from app import crud

router = Router(name="messages-router") 


@router.message()
async def general_handler(message: Message) -> None:
    text = message.text
    try:
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
            curr_user = await crud.user.get_by_tg_chat_id(session, message.chat.id)
        
            messages = curr_user.messages[-settings.USER_HISTORY_LIMIT:]
            # Check if we need to sort here
            #messages = sorted(curr_user.messages, key=lambda x: x.created_at, reverse=True)[:settings.USER_HISTORY_LIMIT]
            print(messages)

            state = build_state([(msg.message_type.value, msg.content) for msg in messages], str(curr_user.tg_chat_id))
            new_state = agent_instance.invoke(state)

            type, answ_text = new_state.history[-1].type, new_state.history[-1].content
            
            # Save bot's response
            await crud.message.create(
                session,
                {
                    "user_id": curr_user.id,
                    "content": answ_text,
                    "message_type": type
                }
            )
            await message.answer(answ_text)

    except Exception as e:
        print(e)
        await message.answer("Internal error")
