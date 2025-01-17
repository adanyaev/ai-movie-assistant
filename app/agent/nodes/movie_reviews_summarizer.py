from pathlib import Path
import requests
import os
import copy

from langchain_core.output_parsers import StrOutputParser, BaseOutputParser
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

from app.agent.nodes._base_api_tool import BaseApiTool
from . import kp_utils


load_dotenv(Path(__file__).parent.parent.parent.parent.resolve() / ".env")


MOVIE_REVIEWS_SUMMARIZE_PROMPT_TEMPLATE = """
## System
Ты — интеллектуальный агент для анализа пользовательских отзывов о фильмах. Твоя задача — принимать отзывы пользователей и формировать краткую и информативную суммаризацию, которая отражает основные темы, эмоции, сильные и слабые стороны фильма, упомянутые в отзывах.  

Вот что нужно учитывать:  
1. **Эмоциональный тон:** Определи общий настрой отзывов (положительный, нейтральный, отрицательный).  
2. **Основные темы:** Обозначь ключевые аспекты фильма, которые чаще всего обсуждаются (например, сюжет, игра актеров, режиссура, визуальные эффекты, музыка).  
3. **Сильные стороны:** Суммируй, что пользователи считают самыми удачными элементами фильма.  
4. **Слабые стороны:** Укажи на аспекты, которые вызвали критику.  
5. **Общие впечатления:** Сделай вывод о том, как фильм воспринимается аудиторией в целом.  

Результат должен быть кратким (3-5 предложений), объективным и информативным, чтобы пользователь мог быстро понять суть отзывов. Если отзывы сильно отличаются по тону или содержанию, отрази это в суммаризации.  

---

Отзывы пользователей о фильме {movie_name}:
```text
{reviews}
```

Твой ответ:
"""


class MovieReviewsSummarizer(BaseApiTool):
    MOVIE_BASE_URL = "https://api.kinopoisk.dev/v1.4/movie/search"
    REVIEW_BASE_URL = "https://api.kinopoisk.dev/v1.4/review"

    def __init__(
        self,
        llm: BaseChatModel,
        answer_prompt: str = MOVIE_REVIEWS_SUMMARIZE_PROMPT_TEMPLATE,
        answer_parser: BaseOutputParser = StrOutputParser(),
        name = "MovieReviewsSummarizer",
        description = "По названию фильма находит самые релевантые отзывы и суммаризирует их",
        limit: int = 5,
        show_logs: bool = False,
    ):
        self._answer_chain = PromptTemplate.from_template(answer_prompt) | llm | answer_parser
        self._description = description
        self._name = name
        self._limit = limit
        self._show_logs = show_logs


    def _invoke(self, question: str, collected_info: str, *args, **kwargs) -> str:

        movie_name = question.strip("\"\' *\n")

        headers = {
            "accept": "application/json",
            "X-API-KEY": os.environ["KP_API_KEY"]
        }
        params = copy.deepcopy(kp_utils.default_search_params)
        params["limit"] = 1
        params["type"] = kp_utils.all_item_types
        params["query"] = movie_name
        api_response = requests.get(self.MOVIE_BASE_URL, params=params, headers=headers)
        if not api_response.ok:
            return "Произошла ошибка при обращении к API"
        movie_id = api_response.json()["docs"][0]['id']

        review_search_params = {
            "page": 1,
            "limit": 100,
            "movieId": [movie_id],
            "sortField": "createdAt",
            "sortType": "-1",
            #"selectFields": ["reviewLikes", "review"]
        }
        review_api_response = requests.get(self.REVIEW_BASE_URL, params=review_search_params, headers=headers)
        if not review_api_response.ok:
            return "Произошла ошибка при обращении к API"
        movie_reviews = sorted(review_api_response.json()["docs"], key=lambda x: x['userRating'], reverse=True)
        movie_reviews = "\n\n---\n\n".join([i['review'] for i in movie_reviews[:self._limit]])

        answer = self._answer_chain.invoke({"movie_name": movie_name, "reviews": movie_reviews})

        if self._show_logs:
            print(f"---{self._name}---")
            print(movie_name)
            print(answer)
            print("-------------------")

        return answer


if __name__ == "__main__":
    from app.agent.llms import LLMFactory

    gpt = LLMFactory.get_llm("deepinfra/Llama-3.3-70B-Instruct")

    summarizer = MovieReviewsSummarizer(gpt, show_logs=True)
    # print(summarizer.invoke("Cубстанция", ""))
    print(summarizer.invoke("Зеленая миля", ""))
