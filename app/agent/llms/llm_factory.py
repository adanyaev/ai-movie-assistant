from typing import Literal
from pathlib import Path
import os

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv


BASE_PATH = Path(__file__).parent.parent.parent.parent.resolve() / ".env"
load_dotenv(BASE_PATH)


class LLMFactory:
    SUPPORTED_MODELS = Literal["gpt-4o", "gpt-4o-mini"]

    _gpt = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        max_tokens=None,
        timeout=6000,
        max_retries=2,
        api_key=os.environ["OPENAI_API_KEY"]
    )

    _gpt_mini = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        max_tokens=None,
        timeout=6000,
        max_retries=2,
        api_key=os.environ["OPENAI_API_KEY"]
    )

    _model_name_to_class = {
        "gpt-4o": _gpt,
        "gpt-4o-mini": _gpt_mini
    }

    @classmethod
    def get_llm(cls, llm_name: SUPPORTED_MODELS) -> BaseChatModel:
        return cls._model_name_to_class[llm_name]
