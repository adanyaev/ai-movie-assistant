from typing import Literal, Dict
from pathlib import Path
import os

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv


BASE_PATH = Path(__file__).parent.parent.parent.parent.resolve() / ".env"
load_dotenv(BASE_PATH)


class LLMFactory:
    SUPPORTED_MODELS = Literal["gpt-4o", "gpt-4o-mini", "deepinfra/Llama-3.3-70B-Instruct"]

    _initialized_models: Dict[str, BaseChatModel] = {}


    @classmethod
    def get_llm(cls, model_name: SUPPORTED_MODELS) -> BaseChatModel:
        if model_name not in cls._initialized_models:
            if model_name == "gpt-4o":
                cls._initialized_models[model_name] = ChatOpenAI(
                    model="gpt-4o",
                    temperature=0,
                    max_tokens=None,
                    timeout=6000,
                    max_retries=2,
                    api_key=os.environ["OPENAI_API_KEY"]
                )
            elif model_name == "gpt-4o-mini":
                cls._initialized_models[model_name] = ChatOpenAI(
                    model="gpt-4o-mini",
                    temperature=0,
                    max_tokens=None,
                    timeout=6000,
                    max_retries=2,
                    api_key=os.environ["OPENAI_API_KEY"]
                )
            elif model_name == "deepinfra/Llama-3.3-70B-Instruct":
                cls._initialized_models[model_name] = ChatOpenAI(
                    model="meta-llama/Llama-3.3-70B-Instruct",
                    base_url="https://api.deepinfra.com/v1/openai",
                    api_key=os.environ["DEEPINFRA_KEY"],
                    temperature=0,
                    max_tokens=None,
                    timeout=6000,
                    max_retries=2,
                )
            else:
                raise ValueError(f"Unsupported model: {model_name}")
        return cls._initialized_models[model_name]
