from aiogram import Router
from aiogram.types import Message
from psycopg import AsyncConnection

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
#from langgraph.checkpoint.memory import MemorySaver

from app.core.config import settings
import app.agent.agent as agent
from app.core.database import async_session_factory
from app import crud

router = Router(name="messages-router") 


@router.message()
async def general_handler(message: Message) -> None:
    async with async_session_factory() as session:
        # Save user's message
        curr_user = await crud.user.get_by_tg_chat_id(session, message.chat.id)
        if not curr_user:
            curr_user = await crud.user.create(session, {"full_name": message.from_user.full_name, "tg_chat_id": message.chat.id})
        await crud.message.create(
            session, 
            {
                "user_id": curr_user.id,
                "content": message.text,
                "message_type": "human"
            }
        )

    async with await AsyncConnection.connect(settings.DB_URI, **agent.connection_kwargs) as conn:
        checkpointer = AsyncPostgresSaver(conn)
        try:
            text = message.text
            app = agent.workflow.compile(checkpointer=checkpointer)
            final_state = await app.ainvoke(
                {"messages": [("human", text)]},
                config={"configurable": {"thread_id": str(message.chat.id)}}
            )
            answ_text = final_state["messages"][-1].content
            
            # Save bot's response
            async with async_session_factory() as session:
                await crud.message.create(
                    session,
                    {
                        "user_id": curr_user.id,
                        "content": answ_text,
                        "message_type": "ai"
                    }
                )
            
            await message.answer(answ_text)
        except Exception as e:
            print(e)
            await message.answer("Internal error")
