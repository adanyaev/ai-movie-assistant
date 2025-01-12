from pathlib import Path
import requests
import os
import copy

from langchain_core.output_parsers import StrOutputParser, BaseOutputParser, JsonOutputParser
from langchain_core.language_models import BaseChatModel
from dotenv import load_dotenv

from app.agent.nodes.planner_node import MOVIES_SEARCH_FIELDS as OUTPUT_FIELDS
from app.agent.nodes._base_api_tool import BaseApiTool
from . import kp_utils

load_dotenv(Path(__file__).parent.parent.parent.parent.resolve() / ".env")


MOVIE_SEARCH_BY_NAME_PROMPT_TEMPLATE = """
## System
Ты помощник в составлении запроса для API с данными о фильмах и сериалах.

## Твоя задача
Тебе на вход приходит вопрос QUESTION и ранее собранная информация COLLECTED_INFO.
Тебе нужно составить словарь параметров для отправки http запроса, чтобы получить ответ на QUESTION.

Твой запрос будет отправлен к API, со следующими полями для поиска:
query (string, required) - Название фильма или сериала

## Требования
1. Составлять запрос можно только с полями, указанными выше.
2. Ответ должен быть в формате словаря, где ключ - название параметра, а значение - его значение. БЕЗ ЛИШНИХ СИМВОЛОВ
3. Учти, что COLLECTED_INFO может быть пустым или не относится к вопросу QUESTION.

## Пример 1
QUESTION:
Когда вышел фильм "Начало"?
COLLECTED_INFO:

Твой ответ:
{{
    "query": "Начало"
}}


QUESTION:
{question}
COLLECTED_INFO:
{collected_info}
Твой ответ:
"""


MOVIE_SEARCH_BY_NAME_ANSWER_PROMPT_TEMPLATE = """
## System
Ты отвечаешь на вопрос пользователя о фильмах и сериалах.

## Твоя задача
Тебе на вход приходит вопрос QUESTION и данные INFO, в которых нужно искать информацию.
Дай ответ на QUESTION, используя данные из INFO. В ответе частично повтори вопрос, чтобы можно было понять что конкретно ты нашел.

## Описание полей в INFO
{fields}

## Пример 1
QUESTION:
Какой рейтинг у фильма "Матрица"?

INFO:
```text
Title: Гладиатор 2
Year: 2024
Type: Movie
Country: Великобритания, США, Марокко, Канада, Мальта
Genres: боевик, драма, приключения, история
Duration: 148 minutes
MPAA Rating: r

Description: 200 год нашей эры. Армия Римской империи под командованием генерала Марка Акация штурмует Нумидию — последнее свободное государство в Северной Африке. В битве с захватчиками у воина Ханно погибает супруга-лучница, а сам он попадает в плен. Вместе с другими пленникам его готовятся продать в рабство, но благодаря физической выносливости и боевым навыкам его замечает и покупает организатор гладиаторских боёв Макрин. Так воин становится гладиатором, одержимым жаждой мести римскому полководцу. Макрин обещает Ханно устроить встречу с его заклятым врагом, если парень будет эффектно, красочно и яростно сражаться на арене Колизея.

Ratings:
- Kinopoisk: 6.321 (38657 votes)
- IMDb: 6.7 (144924 votes)
- Film Critics: 6.6 (373 votes)
- Russian Film Critics: 75
- Awaiting Audience Rating: None
```

Твой ответ:
Рейтинг фильма "Матрица" составляет 8.7 баллов


QUESTION:
{question}

INFO:
```text
{info}
```

Твой ответ:
"""


class MovieSearchByName(BaseApiTool):
    BASE_URL = "https://api.kinopoisk.dev/v1.4/movie/search"

    def __init__(
        self,
        llm: BaseChatModel,
        api_prompt: str = MOVIE_SEARCH_BY_NAME_PROMPT_TEMPLATE,
        answer_prompt: str = MOVIE_SEARCH_BY_NAME_ANSWER_PROMPT_TEMPLATE,
        api_parser: BaseOutputParser = JsonOutputParser(),
        answer_parser: BaseOutputParser = StrOutputParser(),
        name = "MovieSearchByName",
        description = "Возвращает данные о фильме/сериале по его имени",
        limit: int = 1,
        show_logs: bool = False,
    ):
        super().__init__(llm, api_prompt, answer_prompt, api_parser, answer_parser, name, description, limit, show_logs)


    def _invoke(self, question: str, collected_info: str) -> str:
        params_generated = self._chain.invoke({"question": question, "collected_info": collected_info})
        headers = {
            "accept": "application/json",
            "X-API-KEY": os.environ["KP_API_KEY"]
        }
        params = copy.deepcopy(kp_utils.default_search_params)
        params["page"] = 1
        params["limit"] = self._limit
        params["query"] = params_generated["query"]
        api_response = requests.get(self.BASE_URL, params=params, headers=headers)
        if not api_response.ok:
            return "Произошла ошибка при обращении к API"
        api_response = kp_utils.transform_movie_data(api_response.json()["docs"][0]) if api_response else None

        #TODO: maybe add collected_info to api reponse info ?
        api_answer = self._answer_chain.invoke({"fields": OUTPUT_FIELDS, "question": question, "info": api_response})

        if self._show_logs:
            print(f"---{self._name}---")
            print(api_response)
            print(api_answer)
            print("-------------------")

        return api_answer


if __name__ == "__main__":
    from app.agent.llms import LLMFactory

    gpt = LLMFactory.get_llm("deepinfra/Llama-3.3-70B-Instruct")

    search = MovieSearchByName(gpt, show_logs=True)
    print(search.invoke("Какой рейтинг у фильма Ирония судьбы?", ""))
