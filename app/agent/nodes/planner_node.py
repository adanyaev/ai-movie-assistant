import re

from langchain_core.output_parsers import StrOutputParser, BaseOutputParser, JsonOutputParser
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import FunctionMessage
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

from app.agent.nodes._base_node import BaseNode
from app.agent.graph.state import AgentState


PEOPLE_SEARCH_BY_NAME_FIELDS = """
id (number, required) - Уникальный идентификатор персоны на платформе.
name (string, required) - Полное имя персоны.
enName (string, required) - Имя персоны на английском языке.
photo (string, required) - URL фотографии персоны.
sex (string, required) - Пол персоны (например, "Мужской" или "Женский").
birthday (string, required) - Дата рождения персоны в формате "YYYY-MM-DD".
death (string, required) - Дата смерти персоны в формате "YYYY-MM-DD" (если применимо).
age (number, required) - Возраст персоны (или возраст на момент смерти).
birthPlace (array of objects) - Место рождения персоны.
    value (string): Название места рождения.
deathPlace (array of objects) - Место смерти персоны (если применимо).
    value (string): Название места смерти.
profession (array of objects) - Профессии персоны (например, актер, режиссер).
    value (string): Название профессии.
growth (number, required) - Рост персоны (в сантиметрах).
"""

PEOPLE_SEARCH_FIELDS = """
1. id (number, required) - Уникальный идентификатор персоны.
2. name (string | null) - Полное имя персоны.
3. enName (string | null) - Имя персоны на английском языке.
4. photo (string | null) - URL фотографии персоны.
6. sex (string | null) - Пол персоны (например, "Мужской" или "Женский").
7. growth (number | null) - Рост персоны в сантиметрах.
8. birthday (string | null) - Дата рождения персоны в формате "YYYY-MM-DD".
9. death (string | null) - Дата смерти персоны в формате "YYYY-MM-DD" (если применимо).
10. age (number | null) - Возраст персоны (или возраст на момент смерти).
11. birthPlace (array of objects) - Место рождения персоны.
    11.1. value (string): Название места рождения.
12. deathPlace (array of objects) - Место смерти персоны (если применимо).
    12.1. value (string) - Название места смерти.
13. spouses (object) - Информация о супруг(ах).
    13.1. id (number) - Уникальный идентификатор супруга.
    13.2. name (string) - Имя супруга.
    13.3. divorced (boolean) - Статус развода (true, если в разводе).
    13.4. divorcedReason (string) - Причина развода.
    13.5. sex (string) - Пол супруга.
    13.6. children (number) - Количество детей.
    13.7. relation (string) - Тип отношений.
14. countAwards (number) - Количество наград, полученных персоной.
15. profession (array of objects) - Профессии персоны.
    15.1. value (string): Название профессии.
16. facts (array of objects) - Интересные факты о персоне.
    16.1. value (string): Текст факта.
17. movies (array of objects) - Список фильмов, связанных с персоной.
    17.1. id (number, required) - Уникальный идентификатор фильма.
    17.2. name (string | null) - Название фильма.
    17.3. alternativeName (string | null) - Альтернативное название фильма.
    17.4. rating (number | null) - Рейтинг фильма.
    17.5. general (boolean | null) - Указывает, является ли фильм основным в карьере.
    17.6. description (string | null) - Описание фильма.
    17.7. enProfession (string | null) - Профессия персоны.
18. updatedAt (date-time, required) - Дата последнего обновления информации о персоне.
19. createdAt (date-time, required) - Дата создания записи о персоне.
"""


MOVIES_SEARCH_BY_NAME_FIELDS = """
id (number, required) - Уникальный идентификатор фильма.
name (string, required) - Название фильма.
alternativeName (string, required) - Альтернативное название фильма.
enName (string, required) - Название фильма на английском языке.
type (string, required) - Тип контента (например, фильм или сериал).
year (number, required) - Год выхода фильма.
description (string, required) - Полное описание фильма.
shortDescription (string, required) - Краткое описание фильма.
movieLength (number, required) - Продолжительность фильма в минутах.
names (array of objects, required) - Альтернативные названия фильма.
    name (string) - Альтернативное название.
    language (string | null) - Язык названия.
    type (string | null) - Тип названия.
externalId (object | null) - Внешние идентификаторы.
    kpHD (string | null) - Идентификатор из Kinopoisk HD.
    imdb (string | null) - Идентификатор IMDb.
    tmdb (number | null) - Идентификатор TMDb.
logo (object) - Логотип фильма.
    url (string | null) - URL логотипа.
poster (object) - Постер фильма.
backdrop (object) - Фоновое изображение фильма.
rating (object) - Рейтинг фильма.
votes (object) - Информация о голосах за фильм.
    kp (string | null) - Количество голосов на Kinopoisk.
    imdb (number | null) - Количество голосов на IMDb.
    tmdb (number | null) - Количество голосов на TMDb.
    filmCritics (number | null) - Количество голосов кинокритиков.
    russianFilmCritics (number | null) - Количество голосов российских кинокритиков.
    await (number | null) - Количество ожидающих выхода.
genres (array of objects) - Жанры фильма.
    name (string) - Название жанра.
countries (array of objects) - Страны производства.
    name (string) - Название страны.
releaseYears (array of objects) - Годы выхода фильма.
    start (number | null) - Год начала выхода.
    end (number | null) - Год окончания выхода.
isSeries (boolean, required) - Является ли контент сериалом.
ticketsOnSale (boolean, required) - Доступны ли билеты на продажу.
totalSeriesLength (number, required) - Общая продолжительность сериала.
seriesLength (number, required) - Длительность одного эпизода сериала.
ratingMpaa (string, required) - Рейтинг MPAA (возрастное ограничение).
ageRating (number, required) - Возрастное ограничение.
top10 (number | null) - Позиция в топ-10 фильмов.
top250 (number | null) - Позиция в топ-250 фильмов.
typeNumber (number, required) - Числовое представление типа контента.
status (string, required) - Статус фильма.
internalNames (array of strings, required) - Внутренние названия фильма.
internalRating (number, required) - Внутренний рейтинг фильма.
"""


MOVIES_SEARCH_FIELDS = """
id (number, required) - Уникальный идентификатор фильма на Кинопоиске.
externalId (object, required) - Внешние идентификаторы.
    kpHD (string | null) - Идентификатор из Kinopoisk HD.
    imdb (string | null) - Идентификатор IMDb.
    tmdb (number | null) - Идентификатор TMDb.
name (string | null) - Название фильма.
alternativeName (string | null) - Альтернативное название.
enName (string | null) - Название на английском языке.
names (array of objects, required) - Список альтернативных названий.
    name (string) - Название.
    language (string | null) - Язык.
    type (string | null) - Тип названия.
type (string, required) - Тип тайтла (например, movie, tv-series).
typeNumber (number, required) - Тип тайтла в числовом формате (1 для movie, 2 для tv-series и т.д.).
year (number | null) - Год премьеры.
description (string | null) - Описание фильма.
shortDescription (string | null) - Краткое описание фильма.
slogan (string | null) - Слоган фильма.
status (string | null) - Статус релиза (например, filming, completed).
rating (object) - Рейтинг фильма.
    kp (number | null) - Рейтинг на Кинопоиске.
    imdb (number | null) - Рейтинг IMDb.
    tmdb (number | null) - Рейтинг TMDb.
    filmCritics (number | null) - Рейтинг кинокритиков.
    russianFilmCritics (number | null) - Рейтинг российских кинокритиков.
    await (number | null) - Рейтинг ожиданий.
votes (object) - Голоса за фильм.
    kp (string | null) - Голоса на Кинопоиске.
    imdb (number | null) - Голоса на IMDb.
    tmdb (number | null) - Голоса на TMDb.
    filmCritics (number | null) - Голоса кинокритиков.
    russianFilmCritics (number | null) - Голоса российских кинокритиков.
    await (number | null) - Количество ожидающих выхода.
movieLength (number | null) - Продолжительность фильма.
ratingMpaa (string | null) - Возрастной рейтинг MPAA.
ageRating (number | null) - Возрастной рейтинг.
logo (object) - Логотип фильма.
    url (string | null) - URL логотипа.
poster (object) - Постер фильма.
    url (string | null) - URL постера.
    previewUrl (string | null) - URL превью постера.
backdrop (object) - Фоновое изображение.
    url (string | null) - URL фонового изображения.
    previewUrl (string | null) - URL превью фонового изображения.
videos (object) - Видео, связанные с фильмом.
    trailers (array of objects) - Трейлеры.
        url (string | null) - URL трейлера.
        name (string | null) - Название трейлера.
        site (string | null) - Платформа.
        type (string | null) - Тип видео.
        size (number, required) - Размер видео.
    teasers (array of objects, required) - Тизеры.
        url (string | null) - URL тизера.
        name (string | null) - Название тизера.
        site (string | null) - Платформа.
        type (string | null) - Тип видео.
        size (number, required) - Размер видео.
genres (array of objects) - Жанры фильма.
    name (string) - Название жанра.
countries (array of objects) - Страны производства.
    name (string) - Название страны.
persons (array of objects) - Участники съёмочной команды.
    id (number | null) - Идентификатор персоны на Кинопоиске.
    photo (string | null) - URL фотографии.
    name (string | null) - Имя персоны.
    enName (string | null) - Имя на английском.
    description (string, required) - Описание роли.
    profession (string, required) - Профессия.
    enProfession (string, required) - Профессия на английском.
reviewInfo (object) - Информация о рецензиях.
    count (number | null) - Общее количество рецензий.
    positiveCount (number | null) - Количество положительных рецензий.
    percentage (string | null) - Процент положительных рецензий.
seasonsInfo (array of objects) - Информация о сезонах.
    number (number | null) - Номер сезона.
    episodesCount (number | null) - Количество эпизодов.
budget (object) - Бюджет фильма.
    value (number | null) - Сумма.
    currency (string | null) - Валюта.
fees (object) - Сборы фильма.
    world (object) - Сборы в мире.
    russia (object) - Сборы в России.
    usa (object) - Сборы в США.
premiere (object) - Премьера фильма.
    country (string | null) - Страна премьеры.
    world (date-time | null) - Дата мировой премьеры.
    russia (date-time | null) - Дата премьеры в России.
    digital (string | null) - Дата выхода в цифровом формате.
    cinema (date-time | null) - Дата премьеры в кинотеатрах.
    bluray (string, required) - Дата выхода на Blu-ray.
    dvd (string, required) - Дата выхода на DVD.
similarMovies (array of objects) - Схожие фильмы.
    id (number | null) - Идентификатор фильма.
    rating (object, required) - Рейтинги.
        kp (number | null) - Рейтинг Кинопоиска.
        imdb (number | null) - Рейтинг IMDb.
        tmdb (number | null) - Рейтинг TMDb.
        filmCritics (number | null) - Рейтинг кинокритиков.
        russianFilmCritics (number | null) - Рейтинг российских кинокритиков.
        await (number | null) - Рейтинг ожиданий.
year (number, required) - Год выпуска.
name (string, required) - Название фильма.
enName (string, required) - Название на английском.
alternativeName (string, required) - Альтернативное название.
type (string) - Тип тайтла.
poster (object, required) - Постер.
    url (string | null) - URL постера.
    previewUrl (string | null) - URL превью постера.
sequelsAndPrequels (array of objects) - Сиквелы и приквелы.
    id (number | null) - Идентификатор.
    rating (object, required) - Рейтинги.
    year (number, required) - Год выпуска.
    name (string, required) - Название.
    enName (string, required) - Название на английском.
    alternativeName (string, required) - Альтернативное название.
    type (string) - Тип тайтла.
    poster (object, required) - Постер.
watchability (object) - Где можно посмотреть фильм.
    items (array of objects) - Элементы.
releaseYears (array of objects) - Годы выхода.
    start (number | null) - Год начала.
    end (number | null) - Год окончания.
top10 (number | null) - Позиция в топ-10.
top250 (number | null) - Позиция в топ-250.
ticketsOnSale (boolean | null) - Доступность билетов.
totalSeriesLength (number | null) - Общая продолжительность сериала.
seriesLength (number | null) - Длительность серии.
isSeries (boolean, required) - Признак сериала.
audience (array of objects | null) - Информация о зрителях.
lists (array of strings | null) - Коллекции,
в которых находится тайтл.
networks (array of objects, required) - Сетевые платформы или компании, транслирующие фильм.
    items (array of objects, required) - Элементы списка.
        name (string, required) - Название сети.
        logo (object, required) - Логотип.
        url (string | null) - URL логотипа.
updatedAt (date-time, required) - Дата последнего обновления данных о фильме.
createdAt (date-time, required) - Дата создания записи о фильме.
facts (array of objects, required) - Интересные факты о фильме.
    value (string, required) - Текст факта.
    type (string, required) - Тип факта.
    spoiler (boolean, required) - Указывает, содержит ли факт спойлеры.
imagesInfo (object, required) - Информация об изображениях фильма.
    postersCount (number, required) - Количество постеров.
    backdropsCount (number, required) - Количество фонов.
    framesCount (number, required) - Количество кадров.
"""


PLANNER_PROMPT_TEMPLATE = """
## System prompt
Ты помощник пользователя для поиска информации о фильмах, сериалах, режиссерах, актерах и так далее.
И ты составляешь план по ответу на вопрос пользователя, используя других LLM агентов.
Всегда отвечай только на русском языке.

## Твоя задача
Тебе на вход приходит история диалога пользователя (user) и ассистента (assistant) **HISTORY**.
Более поздние сообщения являются более релевантными. Твоя задача составить план чтобы максимально удовлетворить просьбу пользователя и ответить на поставленный вопрос.
Для этого тебе нужно проанализировать диалог и составить пошаговый план по ответу на вопрос.

### Анализ плана
Построй логическую цепочку действий которые нужно выполнить другим агентам и задай вопросы к каждому из выбранных тобой.
Учти, что каждый следующий агент будет использовать информацию, полученную предыдущим.

### Агенты
PeopleSearchByName - агент возвращает информацию о людях, если известны имена этих личностей.
Возвращает список объектов, где каждый объект содержит следующие поля:
{people_search_by_name_fields}

PeopleSearch - агент возвращает информацию о режиссерах, актерах, сценаристах и т.д. если имя неизвестно.
Возвращает список объектов, где каждый объект содержит следующие поля:
{people_search_fields}

MOVIES_SEARCH_BY_NAME - агент возвращает информацию о фильмах, сериалах, аниме и т.д. если известно название.
Возвращает список объектов, где каждый объект содержит следующие поля:
{movies_search_by_name_fields}

MOVIES_SEARCH - агент возвращает информацию о фильмах, сериалах, аниме и т.д.
Возвращает список объектов, где каждый объект содержит следующие поля:
{movies_search_fields}


### Алгоритм построения плана
1. Построй логическую цепочку рассуждений как прийти к ответу.
2. Выбери тех агентов которые могут помочь для поиска ответа.
3. Каждому из этих агентов составь однозначно интерпретируемый вопрос емко и четко.
4. Учти, что все агенты будут выполняться последовательно, поэтому следует учитывать последовательность вызова этих агентов.
5. Учитывай какие поля требуются для ответа на вопрос и вызывай агентов которые их возвращают

## Формат данных
**HISTORY** это список строк из реплик.
Твой ответ должен содержать цепочку рассуждений и итоговый план в блоке ```json<Ответ>```,
где ответ должен быть в следующем формате:
{format_instructions}

## Приступаем
HISTORY:
{history}
Твой ответ:
"""


class AgentTask(BaseModel):
    agent: str = Field(description="Имя агента")
    question: str = Field(description="Вопрос для агента")


class AgentTaskList(BaseModel):
    tasks: list[AgentTask] = Field(description="Итоговая последовательность с вызовом агентов")


class PlannerNode(BaseNode):
    def __init__(
        self,
        llm: BaseChatModel,
        prompt: str = PLANNER_PROMPT_TEMPLATE,
        parser: BaseOutputParser = StrOutputParser(),
        name = "PlannerNode",
        description = "Возвращает план поиска информации",
        show_logs: bool = False,
    ):
        prompt = PromptTemplate(template=prompt, partial_variables={
            "people_search_by_name_fields": PEOPLE_SEARCH_BY_NAME_FIELDS,
            "people_search_fields": PEOPLE_SEARCH_FIELDS,
            "movies_search_by_name_fields": MOVIES_SEARCH_BY_NAME_FIELDS,
            "movies_search_fields": MOVIES_SEARCH_FIELDS,
            "format_instructions": JsonOutputParser(pydantic_object=AgentTaskList).get_format_instructions()
        })
        self._chain = prompt | llm | parser
        self._description = description
        self._name = name
        self._show_logs = show_logs

    def _invoke(self, state: AgentState) -> str:
        history = self._history_to_str(state.history)
        
        response = self._chain.invoke({"history": history})
        pattern = r"`{3}json\s*(.*?)\s*`{3}"
        plan = AgentTaskList.model_validate_json(re.findall(pattern, response, re.DOTALL)[0])
        
        if self._show_logs:
            print(f"---{self._name}---")
            print(plan)
            print("-------------------")

        state.history.append(FunctionMessage(name=self._name, content=response, response_metadata=plan.model_dump()))
        return state


if __name__ == "__main__":
    from langchain_core.messages import HumanMessage
    from app.agent.llms import LLMFactory

    state = AgentState(history=[HumanMessage("Женат ли Киллиан Мерфи?")], user_id="test_user")
    gpt = LLMFactory.get_llm("gpt-4o")

    planner = PlannerNode(gpt)
    result = planner.invoke(state)
    print(result)
