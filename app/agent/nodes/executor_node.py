from typing import List
import re

from langchain_core.output_parsers import StrOutputParser, BaseOutputParser
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage

from app.agent.nodes.people_search_by_name import PeopleSearchByName
from app.agent.nodes.planner_node import AgentTaskList, AgentTask
from app.agent.nodes._base_api_tool import BaseApiTool
from app.agent.nodes.people_search import PeopleSearch
from app.agent.nodes._base_node import BaseNode
from app.agent.graph.state import AgentState


EXECUTOR_PROMPT_TEMPLATE = """
## System
Ты помощник в ответе на вопросы пользователя о фильмах, сериалах, актерах и т.д.

## Твоя задача
Тебе на вход приходит история диалога пользователя и ассистента HISTORY. Последние сообщения пользователя актуальнее.
Также ты получаешь собранную информацию из баз данных COLLECTED_INFO, которую нужно использовать для ответа на вопрос, который
интересует пользователя.

Старайся дать более развернутый ответ. Ничего не придумывай

## HISTORY
{history}
## COLLECTED INFO
{collected_info}
## Твой ответ:
"""


class ExecutorNode(BaseNode):
    def __init__(
        self,
        llm: BaseChatModel,
        executors: List[BaseApiTool],
        prompt: str = EXECUTOR_PROMPT_TEMPLATE,
        parser: BaseOutputParser = StrOutputParser(),
        name="ExecutorNode",
        description="Выполняет все тулы, согласно плану",
        show_logs: bool = False,
    ):
        super().__init__(llm, prompt, parser, name, description, show_logs)
        self.executors = executors
        self._name_to_executor = {executor._name: executor for executor in executors}

    def _format_collected_info_for_prompt(self, collected_info: List[str]) -> str:
        return "\n".join(collected_info)

    def _invoke(self, state: AgentState) -> str:
        collected_info = [self._format_preferences_for_prompt(state.user_preferences)]
        plan: AgentTaskList = AgentTaskList.model_validate(
            state.history[-1].response_metadata
        )

        for task in plan.tasks:

            # TODO: add search of closest executor name
            executor = self._name_to_executor[task.agent]

            collected_info.append(
                executor.invoke(
                    task.question,
                    self._format_collected_info_for_prompt(collected_info),
                    user_id=state.user_id,
                )
            )

        answer = self._chain.invoke(
            {
                "history": self._history_to_str(state.history),
                "collected_info": self._format_collected_info_for_prompt(
                    collected_info
                ),
            }
        )

        if self._show_logs:
            print(f"---{self._name}---")
            print(self._format_collected_info_for_prompt(collected_info))
            print(answer)
            print("-------------------")

        state.history.append(AIMessage(name=self._name, content=answer))
        return state


if __name__ == "__main__":
    from langchain_core.messages import HumanMessage, FunctionMessage
    from app.agent.llms import LLMFactory

    gpt = LLMFactory.get_llm("gpt-4o")

    plan = AgentTaskList(tasks=[
        AgentTask(agent="PeopleSearchByName", question="Верни уникальный идентификатор Киллиана Мерфи"),
        AgentTask(agent="PeopleSearch", question="Какой состав семьи у Киллиана Мерфи?"),
    ])
    executors = [
        PeopleSearchByName(gpt),
        PeopleSearch(gpt)
    ]
    state = AgentState(history=[HumanMessage("Есть ли семья у Киллиана Мерфи?"),
                                FunctionMessage(name="PlannerNode", content="", response_metadata=plan.model_dump())], user_id="test_user")

    executor = ExecutorNode(gpt, executors)
    result = executor.invoke(state)
    print(result)
