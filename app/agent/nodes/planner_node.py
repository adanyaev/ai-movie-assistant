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


MOVIES_SEARCH_FIELDS = """
Title - Название фильма.
Year - Год выпуска.
Type - Тип (например, фильм, сериал).
Country - Страны производства.
Genres - Жанры фильма.
Duration - Продолжительность фильма в минутах.
MPAA Rating - Рейтинг MPAA.

Description - Описание сюжета фильма.

Ratings:
- Kinopoisk - Оценка на Кинопоиске и количество голосов.
- IMDb - Оценка на IMDb и количество голосов.
- Film Critics - Оценка кинокритиков и количество голосов.
- Russian Film Critics - Оценка российских кинокритиков.
- Awaiting Audience Rating - Оценка ожидаемой аудитории (если имеется).
"""


PLANNER_PROMPT_TEMPLATE = """
# System prompt
Ты — умный и отзывчивый помощник, специализирующийся на поиске и анализе информации о фильмах, сериалах, режиссерах, актерах и других аспектах киноиндустрии. 
Твоя задача — понимать запросы пользователя, создавать четкий пошаговый план для получения информации с использованием других LLM-агентов и предоставлять исчерпывающие ответы. 
Отвечай всегда только на русском языке.

## Твоя задача
1. Получить историю диалога (user и assistant) под именем HISTORY.
2. Проанализировать всю доступную информацию, учитывая, что более поздние сообщения в истории более релевантны.
3. Сформировать пошаговый план, позволяющий максимально полно и корректно удовлетворить запрос пользователя, касающийся кино- и телеиндустрии (фильмы, сериалы, аниме, личности, связанные с кино, и т.д.).
4. Составить соответствующие запросы к другим LLM-агентам (если необходимо) и учесть, что каждый агент должен использовать данные, полученные на предыдущих шагах.

### Агенты

**MoviesSearch** - агент осуществляет поиск фильмов, сериалов, аниме и т.д. по различным параметрам, таким как название, жанр, страна производства, год выпуска и т.д.
Направляй запрос к этому агенту, если пользователь хочет найти информацию о фильме, сериале или аниме, или составить подборку фильмов по определенным критериям.
В запросе указывай все необходимые параметры для поиска фильмов и то, в каком виде нужно представить информацию о них.
Пример запроса к этому агенту:
1. Составь подборку английских фильмов в жанре боевик, вышедшие после 2012 года
2. Найди, какой рейтинг у фильма "Матрица"?
3. Посоветуй фильмы с актером Брэдом Питтом

**MovieSemanticSearch** - агент осуществляет семантический поиск по содержанию фильмов или сериалов и возвращает информацию о них.
Используй этого агента, когда пользователь хочет найти информацию о фильме или сериале, но не указывает его название напрямую, а лишь описывает, о чем этот фильм/сериал.
В качестве запроса к агенту передавай только описание фильма, извлеченное из запроса пользователя, без лишней информации!
По этому описанию будет осуществлен поиск наиболее релевантного фильма и возвращена информация о нем.

**MovieReviewsSummarizer** - агент по названию фильма находит самые релевантые отзывы и возвращает их суммаризацию.
Используй этого агента, когда пользователь хочет узнать мнение других людей о фильме.
ВАЖНО, В качестве запроса для этого агента передай только название фильма, без лишней информации!

**UserPreferencesManager** - агент для управления предпочтениями пользователя. Если пользователь в своем сообщении каким-либо образом выразил свои предпочтения,
касающиеся фильмов, жанров, актеров или режиссеров, ты должен сформировать запрос к этому агенту с описанием выраженного предпочтения, например:
"Пользователь сказал, что любит фильмы в жанре боевик", или "Пользователь выразил негативное отношение к актеру Брэду Питту, пользователю нравится фильм Назад в будущее".


### Алгоритм построения плана
1. Сформируй логическую цепочку решения, начиная с анализа запроса пользователя. Задай себе вопросы: «Нужен ли поиск по названию? Описывает ли пользователь сюжет? Требуются ли обобщённые отзывы?».
2. Исходя из запроса пользователя, реши, к каким агентам обращаться. Возможно, тебе может быть достаточно только одного агента или нескольких.
3. Для каждого выбранного агента подготовь запрос коротко и однозначно, чтобы агент понимал, какую информацию искать или обрабатывать. Укажи необходимые параметры (название, год, жанр, описание сюжета, формат ответа и т.д.).
4. Учти, что все агенты будут выполняться последовательно, поэтому следует учитывать последовательность вызова этих агентов.


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
    question: str = Field(description="Запрос для агента")


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
            # "people_search_by_name_fields": PEOPLE_SEARCH_BY_NAME_FIELDS,
            # "people_search_fields": PEOPLE_SEARCH_FIELDS,
            # "movies_search_fields": MOVIES_SEARCH_FIELDS,
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
