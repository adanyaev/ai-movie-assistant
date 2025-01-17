from abc import ABC, abstractmethod
from typing import List

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, FunctionMessage
from langchain_core.output_parsers import StrOutputParser, BaseOutputParser
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import PromptTemplate

from app.models.user import PreferenceItem, PreferenceType
from app.schemas.user import UserPreferenceBase
from ..graph import AgentState


class BaseNode(ABC):
    """Base Node class"""
    def __init__(
        self,
        llm: BaseChatModel,
        prompt: str,
        parser: BaseOutputParser = StrOutputParser(),
        name: str | None = None,
        description: str | None = None,
        show_logs: bool = False,
    ) -> None:
        self._chain = PromptTemplate.from_template(prompt) | llm | parser
        self._description = description
        self._name = name
        self._show_logs = show_logs

    def _history_to_str(self, history: List[BaseMessage]):
        roles_to_str = {
            AIMessage: "assistant",
            HumanMessage: "user"
        }
        return "\n".join([f"{roles_to_str[type(x)]}: {x.content}" for x in history
                         if not isinstance(x, FunctionMessage)])

    def _format_preferences_for_prompt(
        self, preferences: List[UserPreferenceBase]
    ) -> str:
        """
        Преобразует данные о предпочтениях пользователя в строку для системного промпта LLM-агента.

        :param preferences: Объект UserPreferences с предпочтениями пользователя
        :return: Строка для использования в системном промпте
        """
        type2russian = {
            PreferenceItem.MOVIE: "фильм",
            PreferenceItem.GENRE: "жанр",
            PreferenceItem.DIRECTOR: "режиссёр",
            PreferenceItem.ACTOR: "актёр",
        }
        formatted_items = []

        for pref in preferences:
            action = (
                "нравится"
                if pref.preference_type == PreferenceType.LIKE
                else "не нравится"
            )
            item_type = type2russian[pref.preference_item]
            formatted_items.append(
                f'Пользователю {action} {item_type} "{pref.item_name}".'
            )

        return "\n".join(formatted_items)

    def invoke(self, state: AgentState) -> str:
        return self._invoke(state)

    @abstractmethod
    def _invoke(self, state: AgentState) -> str:
        pass
