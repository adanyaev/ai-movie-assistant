from abc import ABC, abstractmethod

from langchain_core.output_parsers import StrOutputParser, BaseOutputParser
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import PromptTemplate

from app.graph.state import AgentState


class BaseNode(ABC):
    """Base Node class"""
    def __init__(
        self,
        llm: BaseChatModel,
        prompt: str,
        parser: BaseOutputParser = StrOutputParser(),
        name: str | None = None,
        description: str | None = None
    ) -> None:
        self._chain = PromptTemplate.from_template(prompt) | llm | parser
        self._description = description
        self._name = name

    def invoke(self, state: AgentState) -> str:
        return self._invoke(state)

    @abstractmethod
    def _invoke(self, state: AgentState) -> str:
        pass
