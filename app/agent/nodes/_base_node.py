from abc import ABC, abstractmethod
from typing import List

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, FunctionMessage
from langchain_core.output_parsers import StrOutputParser, BaseOutputParser
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import PromptTemplate

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

    def invoke(self, state: AgentState) -> str:
        return self._invoke(state)

    @abstractmethod
    def _invoke(self, state: AgentState) -> str:
        pass
