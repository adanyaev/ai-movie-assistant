from typing import List

from langchain_core.prompts.chat import _convert_to_message

from app.core.config import settings
from app.schemas.user import UserPreferenceBase
from .llms import LLMFactory
from .graph.movie_agent import MovieAgent
from .graph.state import AgentState


def build_state(messages: list[tuple[str, str]], user_id: int, user_preferences: List[UserPreferenceBase]):
    state = AgentState(
        history=[_convert_to_message(message).format() for message in messages],
        user_id=user_id,
        user_preferences=user_preferences
    )
    return state

llm = LLMFactory.get_llm(settings.LLM_NAME)
agent_instance = MovieAgent(llm=llm, show_logs=settings.VERBOSE_AGENT)
