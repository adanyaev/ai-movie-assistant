from pathlib import Path
import requests
import os
import copy
import random

from aiogram import Bot as TelegramBot
from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession
from langchain_core.output_parsers import StrOutputParser, BaseOutputParser
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

from app.models.user import User, PreferenceItem, PreferenceType
from app.schemas.user import User as UserSchema, UserPreference as UserPreferenceSchema
from app.agent.nodes._base_node import BaseNode
from . import kp_utils


load_dotenv(Path(__file__).parent.parent.parent.parent.resolve() / ".env")


MOVIES_RECOMMEND_PROMPT_TEMPLATE = """
Ты — персональный ассистент по подбору фильмов и сериалов. Твоя задача — помогать пользователю находить кино, которое ему понравится, на основе его предпочтений. Ты дружелюбный, внимательный и заинтересованный в том, чтобы пользователь получил удовольствие от просмотра. Ты умеешь анализировать предпочтения пользователя (жанры, актеры, режиссеры, темы) и составлять персонализированные рекомендации с обоснованием.

**Твоя цель:**  
Сделать процесс выбора фильма увлекательным и персонализированным, чтобы пользователь чувствовал, что его вкусы и предпочтения учтены.

**Инструкции:**  
1. Ты получаешь список фильмов RECOMMENDATIONS, которые были отобраны для пользователя на основе его предпочтений. Для каждого фильма указано, какое именно предпочтение пользователя стало причиной рекомендации (например, любимый фильм, жанр, актер или режиссер).
2. Твоя задача — составить сообщение для пользователя, в котором ты представишь подборку фильмов, объяснив, почему каждый из них был выбран.  
3. Сообщение должно быть дружелюбным, увлекательным и персонализированным.
4. Если есть возможность, добавь краткое описание фильма и рейтинги фильма на разных площадках.
5. В конце предложи пользователю выбрать фильм для просмотра или задать уточняющие вопросы, если он хочет получить больше рекомендаций.

**Пример сообщения:**  
"Привет! Я подобрал для тебя несколько фильмов, которые, думаю, тебе понравятся:  
1. **«Начало» (2010)** — я выбрал его, потому что ты упоминал, что любишь сложные сюжеты и научную фантастику. Этот фильм Кристофера Нолана рассказывает о мире снов и реальности, где герои попадают в чужие сны, чтобы украсть идеи.  
2. **«Криминальное чтиво» (1994)** — я добавил его в подборку, потому что ты говорил, что тебе нравятся фильмы Квентина Тарантино. Это культовая история о преступниках, наполненная остроумными диалогами и неожиданными поворотами.  
3. **«Остров проклятых» (2010)** — я выбрал его, потому что ты любишь триллеры с Леонардо ДиКаприо. Этот фильм — захватывающая история о расследовании таинственных событий на закрытом острове для душевнобольных преступников.

Какой фильм тебе больше всего заинтересовал? Или, может быть, ты хочешь, чтобы я подобрал что-то еще?"

---

RECOMMENDATIONS:
```
{recommendations}
```

Твой ответ:
"""


class RecommendUsersAutonomousTask:

    def __init__(
        self,
        llm: BaseChatModel,
        answer_prompt: str = MOVIES_RECOMMEND_PROMPT_TEMPLATE,
        answer_parser: BaseOutputParser = StrOutputParser(),
        name="RecommendUsersAutonomousTask",
        description="Отправляет всем активным пользователям персональные рекомендации по фильмам",
        limit: int = 5,
        show_logs: bool = False,
    ):
        self._answer_chain = (
            PromptTemplate.from_template(answer_prompt) | llm | answer_parser
        )
        self._description = description
        self._name = name
        self._limit = limit
        self._show_logs = show_logs

    def _preference_to_prompt(
        self, pref: UserPreferenceSchema
    ) -> str:
        """
        Преобразует данные о предпочтениях пользователя в строку для системного промпта LLM-агента.

        :param pref: Объект UserPreference с предпочтениями пользователя
        :return: Строка для использования в системном промпте
        """
        type2russian = {
            PreferenceItem.MOVIE: "фильм",
            PreferenceItem.GENRE: "жанр",
            PreferenceItem.DIRECTOR: "режиссёр",
            PreferenceItem.ACTOR: "актёр",
        }

        action = (
            "нравится"
            if pref.preference_type == PreferenceType.LIKE
            else "не нравится"
        )
        item_type = type2russian[pref.preference_item]
        result = (
            f'Пользователю {action} {item_type} "{pref.item_name}".'
        )

        return result


    def _get_popular_movies_recommendation(self) -> list:
        params = copy.deepcopy(kp_utils.DEFAULT_SEARCH_PARAMS)
        params["lists"] = ["top250"]
        params["page"] = random.randint(1, 250 // self._limit)
        params["limit"] = self._limit
        api_response = requests.get(
            kp_utils.MOVIE_SEARCH_URL, params=params, headers=kp_utils.HEADERS
        )
        return api_response.json()["docs"]

    def _get_personalized_movies_recommendation(
        self,
        positive_prefs: list[UserPreferenceSchema],
        watched_movies: list[int],
        search_limit: int = 50,
    ) -> tuple[list, list]:
        random.shuffle(positive_prefs)
        docs = []
        source_prefs = []
        i = 0
        while i < self._limit:
            if not positive_prefs:
                break
            pref = positive_prefs[i % len(positive_prefs)]
            if pref.preference_item == PreferenceItem.MOVIE:
                api_response = requests.get(
                    kp_utils.MOVIE_SEARCH_URL + f"/{pref.kp_id}", headers=kp_utils.HEADERS
                )
                if not api_response.ok:
                    positive_prefs.pop(i % len(positive_prefs))
                    continue
                api_response = api_response.json()
                if "similarMovies" not in api_response:
                    positive_prefs.pop(i % len(positive_prefs))
                    continue
                pref_docs = []
                for movie in api_response["similarMovies"]:
                    if movie["id"] not in watched_movies:
                        pref_docs.append(requests.get(kp_utils.MOVIE_SEARCH_URL + f"/{movie['id']}", headers=kp_utils.HEADERS).json())

            else:
                if pref.preference_item == PreferenceItem.GENRE:
                    param_key = "genres.name"
                    param_value = [pref.item_name]
                elif (
                    pref.preference_item == PreferenceItem.ACTOR
                    or pref.preference_item == PreferenceItem.DIRECTOR
                ):
                    param_key = "persons.id"
                    param_value = [pref.kp_id]
                else:
                    raise ValueError("Unknown preference item")

                params = copy.deepcopy(kp_utils.DEFAULT_SEARCH_PARAMS)
                params[param_key] = param_value
                params["limit"] = search_limit
                api_response = requests.get(
                    kp_utils.MOVIE_SEARCH_URL, params=params, headers=kp_utils.HEADERS
                )
                if not api_response.ok:
                    positive_prefs.pop(i % len(positive_prefs))
                    continue
                pref_docs = api_response.json()["docs"]
            random.shuffle(pref_docs)
            pref_doc_idx = 0
            while pref_doc_idx < len(pref_docs) and pref_docs[pref_doc_idx]["id"] in watched_movies:
                pref_doc_idx += 1
            if pref_doc_idx < len(pref_docs):
                docs.append(pref_docs[pref_doc_idx])
                source_prefs.append(pref)
                watched_movies.append(pref_docs[pref_doc_idx]["id"])
                i += 1
                continue
            
            positive_prefs.pop(i % len(positive_prefs))

        return docs, source_prefs


    def _prepare_answer(self, movies_docs: list, source_prefs: list | None) -> str:
        movies_data = [kp_utils.transform_movie_data(i) for i in movies_docs]

        recs_str = ""

        for i in range(len(movies_data)):
            recs_str = recs_str + f"## Рекомендация {i + 1}.\n"
            if source_prefs:
                recs_str = recs_str + f"### Причина рекомендации: {self._preference_to_prompt(source_prefs[i])}\n"
            recs_str = recs_str + f"### Информация о рекомендованном фильме:\n{movies_data[i]}\n"
            recs_str = recs_str + "\n\n"

        if self._show_logs:
            print(f"Movies recs:\n{recs_str}")

        return self._answer_chain.invoke({"recommendations": recs_str})

    async def _ainvoke(
        self,
        session: AsyncSession,
        bot: TelegramBot,
        tg_user_id: int = None,
    ):
        if self._show_logs:
            print(f"---{self._name}---")
            print("Starting autonomous task...")

        stmt = select(User).where(User.is_active == True)
        if tg_user_id:
            stmt = stmt.where(User.tg_chat_id == tg_user_id)
        result = await session.execute(stmt)
        users = result.scalars().all()
        users = [UserSchema.model_validate(user) for user in users]

        if self._show_logs:
            print(f"Found {len(users)} active users.")

        for user in users:
            positive_prefs = [
                i for i in user.preferences if i.preference_type == PreferenceType.LIKE
            ]
            source_prefs = None # source preferences of personalized recommendations; None if popular movies recommended
            if not positive_prefs:
                rec_docs = self._get_popular_movies_recommendation()
            else:
                watched_movies = [
                    i.kp_id
                    for i in user.preferences
                    if i.preference_item == PreferenceItem.MOVIE
                ]
                rec_docs, source_prefs = self._get_personalized_movies_recommendation(
                    positive_prefs, watched_movies
                )
                if not rec_docs:
                    rec_docs = self._get_popular_movies_recommendation()

            if self._show_logs:
                print(f"Made up {len(rec_docs)} recommendations for user {user.tg_chat_id}.")

            answer = self._prepare_answer(rec_docs, source_prefs)

            if self._show_logs:
                print(f"Answer for user {user.tg_chat_id}:\n{answer}")
                print("-------------------")

            await bot.send_message(user.tg_chat_id, answer)

        return

    async def ainvoke(self, *args, **kwargs):
        return await self._ainvoke(*args, **kwargs)


if __name__ == "__main__":
    pass
