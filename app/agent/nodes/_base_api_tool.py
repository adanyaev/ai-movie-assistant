from abc import ABC, abstractmethod
from typing import List

from langchain_core.output_parsers import StrOutputParser, BaseOutputParser, JsonOutputParser
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, FunctionMessage
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import PromptTemplate


class BaseApiTool(ABC):
    """Base Tool class"""
    def __init__(
        self,
        llm: BaseChatModel,
        api_prompt: str,
        answer_prompt: str,
        api_parser: BaseOutputParser = JsonOutputParser(),
        answer_parser: BaseOutputParser = StrOutputParser(),
        name: str | None = None,
        description: str | None = None,
        limit: int = 10,
        show_logs: bool = False,
    ) -> None:
        self._chain = PromptTemplate.from_template(api_prompt) | llm | api_parser
        self._answer_chain = PromptTemplate.from_template(answer_prompt) | llm | answer_parser
        self._description = description
        self._name = name
        self._limit = limit
        self._show_logs = show_logs

    def invoke(self, question: str, collected_info: str, *args, **kwargs) -> str:
        return self._invoke(question, collected_info, *args, **kwargs)

    @abstractmethod
    def _invoke(self, question: str, collected_info: str, *args, **kwargs) -> str:
        pass
