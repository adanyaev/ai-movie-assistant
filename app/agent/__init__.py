from langchain_core.prompts.chat import _convert_to_message

from app.core.config import settings

from .llms import LLMFactory
from .graph.movie_agent import MovieAgent
from .graph.state import AgentState


def build_state(messages: list[tuple[str, str]], user_id: str):
    print([_convert_to_message(message).format() for message in messages])
    state = AgentState(
        history=[_convert_to_message(message).format() for message in messages],
        user_id=user_id
    )
    return state

llm = LLMFactory.get_llm(settings.LLM_NAME)
agent_instance = MovieAgent(llm=llm, show_logs=settings.VERBOSE_AGENT)
