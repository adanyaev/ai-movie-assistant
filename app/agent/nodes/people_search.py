from typing import Dict, List
from pathlib import Path
import requests
import os

from langchain_core.output_parsers import StrOutputParser, BaseOutputParser, JsonOutputParser
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from app.agent.nodes.planner_node import PEOPLE_SEARCH_FIELDS as OUTPUT_FIELDS
from app.agent.nodes._base_api_tool import BaseApiTool


load_dotenv(Path(__file__).parent.parent.parent.parent.resolve() / ".env")


PEOPLE_SEARCH_PROMPT_TEMPLATE = """
## System
Ты помощник в составлении запроса для API с данными людей (актеров, режиссеров и т.д.).

## Твоя задача
Тебе на вход приходит вопрос QUESTION и ранее собранная информация COLLECTED_INFO.
Тебе нужно составить словарь параметров для отправки http запроса в формате библиотеки requests.
Чтобы получить ответ на QUESTION. 

Твой запрос будет отправлен к API, со следующими полями для поиска:
"id" - идентификатор человека
"movies.id" - array of strings | null - Поиск по ID фильма (пример: "666", "555", "!666")
"sex" - array of strings | null - Поиск по гендеру (пример: Женский, Мужской)
"growth" - array of strings | null - Поиск по росту (пример: 170-180, 180)
"birthday" - array of strings | null - Поиск по дате рождения (пример: 01.01.2000-01.01.2001, 01.01.2000)
"death" - array of strings | null - Поиск по дате смерти (пример: 01.01.2000-01.01.2001, 01.01.2000)
"age" - array of strings | null - Поиск по возрасту (пример: 18-25, 25)
"birthPlace.value" - array of strings | null - Поиск по месту рождения (пример: Москва, Санкт-Петербург)
"deathPlace.value" - array of strings | null - Поиск по месту смерти (пример: Москва, Санкт-Петербург)
"spouses.id" - array of strings | null - Поиск по ID супруги (пример: "666", "555", "!666")
"children.id" - array of strings | null - Поиск по ID детей (пример: "666", "555", "!666")
"spouses.divorced" - string | null - Поиск по статусу развода (пример: true, false)
"spouses.sex" - array of strings | null - Поиск по гендеру супруги(супруга) (пример: Женский, Мужской)
"countAwards" - array of strings | null - Поиск по количеству наград (пример: 1-10, 10)
"profession.value" - array of strings | null - Поиск по профессии (пример: Актер, Режиссер)
"movies.rating" - array of strings | null - Поиск по рейтингу фильма (пример: 1-10, 10)
"movies.enProfession" - array of strings | null - Поиск по профессии в фильмах на английском (пример: actor, director)
"updatedAt" - array of strings | null - Поиск по дате обновления в базе (пример: 01.01.2020, 01.01.2020-31.12.2020)
"createdAt" - array of strings | null - Поиск по дате создания в базе (пример: 01.01.2020, 01.01.2020-31.12.2020)

## Требования
1. Составлять запрос можно только с полями, указанными выше.
2. Ответ должен быть в формате словаря, где ключ - название параметра, а значение - его значение. БЕЗ ЛИШНИХ СИМВОЛОВ
3. Учти, что COLLECTED_INFO может быть пустым или не относится к вопросу QUESTION.
4. Всегда возвращай поле "name" на всякий случай.
5. Формат твоего ответа:
{format_instructions}

## Список полей которые возвращает запрос
Учти, что "fields" могут быть только эти поля, без вложенных:
{fields}

## Пример 1
QUESTION:
Сколько лет Киану Ривзу?
COLLECTED_INFO:
Идентификатор Киану Ривза - 7836
Твой ответ:
{{
    "params": {{
        "id": "7836"
    }}
    "fields": ["age", "name"]
}}

## Пример 2
QUESTION:
Женат ли Киллиан Мерфи?
COLLECTED_INFO:
Идентификатор Киллиана Мерфи - 5005
Твой ответ:
{{
    "params": {{
        "id": "5005"
    }}
    "fields": ["spouses", "name"]
}}

QUESTION:
{question}
COLLECTED_INFO:
{collected_info}
Твой ответ:
"""


PEOPLE_SEARCH_ANSWER_PROMPT_TEMPLATE = """
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


class ApiOutput(BaseModel):
    params: Dict[str, str] = Field(description="Словарь параметров для запроса, где ключ - название параметра, значение - его значение")
    fields: List[str] = Field(description="Список полей, которые нужно вернуть из запроса (НЕЛЬЗЯ ИСПОЛЬЗОВАТЬ ВЛОЖЕННЫЕ)")


class PeopleSearch(BaseApiTool):
    BASE_URL = "https://api.kinopoisk.dev/v1.4/person"

    def __init__(
        self,
        llm: BaseChatModel,
        api_prompt: str = PEOPLE_SEARCH_PROMPT_TEMPLATE,
        answer_prompt: str = PEOPLE_SEARCH_ANSWER_PROMPT_TEMPLATE,
        api_parser: BaseOutputParser = JsonOutputParser(pydantic_object=ApiOutput),
        answer_parser: BaseOutputParser = StrOutputParser(),
        name = "PeopleSearch",
        description = "Возвращает данные о человеке",
        limit: int = 10,
        show_logs: bool = False,
    ):
        super().__init__(llm, api_prompt, answer_prompt, api_parser, answer_parser, name, description, limit, show_logs)
        self._format_instructions = api_parser.get_format_instructions()

    def _invoke(self, question: str, collected_info: str, *args, **kwargs) -> str:
        request_data = self._chain.invoke({"question": question, "collected_info": collected_info, "fields": OUTPUT_FIELDS,
                                           "format_instructions": self._format_instructions})
        headers = {
            "accept": "application/json",
            "X-API-KEY": os.environ["KP_API_KEY"]
        }
        params = request_data["params"]
        params["page"] = 1
        params["limit"] = self._limit
        params["selectFields"] = request_data["fields"]
        api_response = requests.get(PeopleSearch.BASE_URL, params=params, headers=headers).json()["docs"]
        api_answer = self._answer_chain.invoke({"fields": OUTPUT_FIELDS, "question": question, "info": api_response})

        if self._show_logs:
            print(f"---{self._name}---")
            print(api_response)
            print(api_answer)
            print("-------------------")

        return api_answer


if __name__ == "__main__":
    from app.agent.llms import LLMFactory

    gpt = LLMFactory.get_llm("gpt-4o")

    search = PeopleSearch(gpt)
    print(search.invoke("Сколько лет Киллиану Мерфи?", "Идентификатор Киллиана Мерфи - 5005"))
