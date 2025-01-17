from pathlib import Path
import requests
import os

from langchain_core.output_parsers import StrOutputParser, BaseOutputParser, JsonOutputParser
from langchain_core.language_models import BaseChatModel
from dotenv import load_dotenv

from app.agent.nodes.planner_node import PEOPLE_SEARCH_BY_NAME_FIELDS as OUTPUT_FIELDS
from app.agent.nodes._base_api_tool import BaseApiTool
from . import kp_utils

load_dotenv(Path(__file__).parent.parent.parent.parent.resolve() / ".env")


PEOPLE_SEARCH_BY_NAME_PROMPT_TEMPLATE = """
## System
Ты помощник в составлении запроса для API с данными людей (актеров, режиссеров и т.д.).

## Твоя задача
Тебе на вход приходит вопрос QUESTION и ранее собранная информация COLLECTED_INFO.
Тебе нужно составить словарь параметров для отправки http запроса в формате библиотеки requests.
Чтобы получить ответ на QUESTION. 

Твой запрос будет отправлен к API, со следующими полями для поиска:
query (string, required) - Имя актера, режиссера или другого человека

## Требования
1. Составлять запрос можно только с полями, указанными выше.
2. Ответ должен быть в формате словаря, где ключ - название параметра, а значение - его значение. БЕЗ ЛИШНИХ СИМВОЛОВ
3. Учти, что COLLECTED_INFO может быть пустым или не относится к вопросу QUESTION.

## Пример 1
QUESTION:
Когда родился Кристофер Нолан?
COLLECTED_INFO:

Твой ответ:
{{
    "query": "Кристофер Нолан"
}}

## Пример 2
QUESTION:
Уникальный идентификатор Бредда Питта и Леонардо Ди-Каприо
Твой ответ:
{{
    "query": ["Бредд Питт", "Леонардо Ди-Каприо"]
}}

QUESTION:
{question}
COLLECTED_INFO:
{collected_info}
Твой ответ:
"""


PEOPLE_SEARCH_BY_NAME_ANSWER_PROMPT_TEMPLATE = """
## System
Ты отвечаешь на вопрос пользователя по актерам, режиссерам и т.д.

## Твоя задача
Тебе на вход приходит вопрос QUESTION и данные в формате JSON в которых нужно искать информацию INFO.
Дай ответ на QUESTION, используя данные из INFO. В ответе частично повтори вопрос, чтобы можно было понять что конкретно ты нашел.

## Описание полей в INFO
{fields}

## Пример 1
QUESTION:
Какой рост у Киану Ривза?
INFO:
[
    {{
        "id": 7836,
        "name": "Киану Ривз",
        "growth": 186
    }},
    {{
        "id": 1313,
        "name": "Риз Кан",
        "growth": 174
    }}
]
Твой ответ:
Рост Киану Ривза 186 сантиметров

QUESTION:
{question}
INFO:
{info}
Твой ответ:
"""


class PeopleSearchByName(BaseApiTool):
    BASE_URL = "https://api.kinopoisk.dev/v1.4/person/search"

    def __init__(
        self,
        llm: BaseChatModel,
        api_prompt: str = PEOPLE_SEARCH_BY_NAME_PROMPT_TEMPLATE,
        answer_prompt: str = PEOPLE_SEARCH_BY_NAME_ANSWER_PROMPT_TEMPLATE,
        api_parser: BaseOutputParser = JsonOutputParser(),
        answer_parser: BaseOutputParser = StrOutputParser(),
        name = "PeopleSearchByName",
        description = "Возвращает данные о человеке по его имени",
        limit: int = 5,
        show_logs: bool = False,
        load_info_from_wiki: bool = True
    ):
        super().__init__(llm, api_prompt, answer_prompt, api_parser, answer_parser, name, description, limit, show_logs)
        self._load_info_from_wiki = load_info_from_wiki

    def _invoke(self, question: str, collected_info: str, *args, **kwargs) -> str:

        params = self._chain.invoke({"question": question, "collected_info": collected_info})

        if self._load_info_from_wiki:
            api_response = kp_utils.get_person_info_from_wiki(params["query"])
            if not api_response:
                api_response = "Информация о данном человеке не найдена, или он не относится к киноиндустрии"
            fields = "Информация о человеке со страницы в Википедии"
        else:
            params["page"] = 1
            params["limit"] = self._limit
            api_response = requests.get(PeopleSearchByName.BASE_URL, params=params, headers=kp_utils.headers).json()["docs"]
            fields = OUTPUT_FIELDS
            
        api_answer = self._answer_chain.invoke({"fields": fields, "question": question, "info": api_response})

        if self._show_logs:
            print(f"---{self._name}---")
            print(api_response)
            print(api_answer)
            print("-------------------")

        return api_answer


if __name__ == "__main__":
    from app.agent.llms import LLMFactory

    gpt = LLMFactory.get_llm("gpt-4o")

    search = PeopleSearchByName(gpt)
    print(search.invoke("Сколько лет Киану Ривзу?", ""))
