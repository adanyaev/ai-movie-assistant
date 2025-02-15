from typing import Literal
from pathlib import Path

from pydantic import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser, BaseOutputParser, PydanticOutputParser
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

from app.schemas.user import UserPreferenceBase
from app.models.user import UserPreference as UserPreferenceModel, PreferenceItem, PreferenceType
from app.core.database import engine
from app.agent.nodes._base_api_tool import BaseApiTool
from . import kp_utils


load_dotenv(Path(__file__).parent.parent.parent.parent.resolve() / ".env")

class UserPreferences(BaseModel):
    preferences: list[UserPreferenceBase] = Field(description="Список предпочтений пользователя")


USER_PREFERENCES_MANAGER_PROMPT_TEMPLATE = """
Вы - интеллектуальный ассистент, специализирующийся на анализе предпочтений пользователей в области кино и сериалов. Ваша основная задача - извлекать из сообщений пользователя информацию о его предпочтениях и структурировать её в формате JSON.

ОСНОВНЫЕ ПРИНЦИПЫ:
1. Анализируйте текст пользователя на предмет упоминания:
   - конкретных фильмов или сериалов
   - жанров
   - режиссёров
   - актёров

2. Определяйте тональность отношения:
   - Позитивные выражения ("нравится", "люблю", "отличный", "хороший") -> LIKE
   - Негативные выражения ("не нравится", "ненавижу", "плохой") -> DISLIKE

3. Классифицируйте каждый элемент по типу:
   - MOVIE: конкретные фильмы/сериалы
   - GENRE: жанры (боевик, комедия, драма и т.д.)
   - DIRECTOR: режиссёры
   - ACTOR: актёры

ПРАВИЛА ОБРАБОТКИ:
1. Сохраняйте оригинальное написание имён и названий
2. При неполных или неясных данных пропускайте соответствующий элемент
3. При невозможности определить тип предпочтения (like/dislike) пропускайте элемент
4. При невозможности отнести элемент к одному из типов ("movie" | "genre" | "director" | "actor") пропускайте элемент


ФОРМАТ ВЫВОДА:
Всегда возвращайте результат в формате JSON, соответствующем следующей схеме:
{{
    "preferences": [
        {{
            "item_name": string,
            "preference_item": "movie" | "genre" | "director" | "actor",
            "preference_type": "like" | "dislike"
        }}
    ]
}}

ПРИМЕРЫ:

Входное сообщение:
"Пользователь сказал, что любит фильмы в жанре боевик"

Ответ:
{{
    "preferences": [
        {{
            "item_name": "боевик",
            "preference_item": "genre",
            "preference_type": "like"
        }}
    ]
}}

Входное сообщение:
"Пользователь выразил негативное отношение к актеру Брэду Питту, пользователю нравится фильм Назад в будущее"

Ответ:
{{
    "preferences": [
        {{
            "item_name": "Брэд Питт",
            "preference_item": "actor",
            "preference_type": "dislike"
        }},
        {{
            "item_name": "Назад в будущее",
            "preference_item": "movie",
            "preference_type": "like"
        }}
    ]
}}

---

Входное сообщение:
"{query}"

Твой ответ:
"""


class UserPreferencesManager(BaseApiTool):

    def __init__(
        self,
        llm: BaseChatModel,
        prompt: str = USER_PREFERENCES_MANAGER_PROMPT_TEMPLATE,
        parser: BaseOutputParser = PydanticOutputParser(pydantic_object=UserPreferences),
        name = "UserPreferencesManager",
        description = "По сообщению о предпочтениях пользователя возвращает структурированный список предпочтений",
        show_logs: bool = False,
    ):
        self._chain = PromptTemplate.from_template(prompt) | llm | parser
        self._description = description
        self._name = name
        self._show_logs = show_logs


    def _invoke(self, query: str, collected_info: str, user_id: int, *args, **kwargs) -> str:

        query = query.strip("\"\' *\n")
        prefs = self._chain.invoke({"query": query}).preferences

        infered_prefs = []
        infered_kp_ids = []
        for pref in prefs:
            
            kp_id = None
            if pref.preference_item == PreferenceItem.MOVIE:
                kp_id = kp_utils.InferKpId.movie(pref.item_name)
            elif pref.preference_item == PreferenceItem.GENRE:
                kp_id = kp_utils.InferKpId.genre(pref.item_name)
            elif pref.preference_item == PreferenceItem.ACTOR or pref.preference_item == PreferenceItem.DIRECTOR:
                kp_id = kp_utils.InferKpId.person(pref.item_name)

            if kp_id:
                infered_kp_ids.append(kp_id)
                infered_prefs.append(pref)

        if not infered_prefs:
            return ""

        with engine.connect() as conn:
            stmt = UserPreferenceModel.__table__.insert().values([
                {
                    "item_name": pref.item_name,
                    "preference_item": pref.preference_item,
                    "preference_type": pref.preference_type,
                    "kp_id": infered_kp_ids[i],
                    "user_id": user_id
                } for i, pref in enumerate(infered_prefs)
            ])
            conn.execute(stmt)
            conn.commit()

        if self._show_logs:
            print(f"---{self._name}---")
            print(query)
            print(prefs)
            print("-------------------")

        return ""


if __name__ == "__main__":
    from app.agent.llms import LLMFactory

    gpt = LLMFactory.get_llm("deepinfra/Llama-3.3-70B-Instruct")

    summarizer = UserPreferencesManager(gpt, show_logs=True)
    print(summarizer.invoke("Я люблю мюзиклы", ""))
