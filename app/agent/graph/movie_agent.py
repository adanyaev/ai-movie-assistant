from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph

from app.agent.graph.state import AgentState
from app.agent.nodes import (
    PlannerNode,
    PeopleSearch,
    PeopleSearchByName,
    ExecutorNode
)


class MovieAgent:
    def __init__(
        self,
        llm: BaseChatModel,
        show_logs: bool = False,
        **kwargs,
    ):
        self._llm = llm
        self._show_logs = show_logs
        self._graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(AgentState)

        # tools
        people_search = PeopleSearch(self._llm, show_logs=self._show_logs)
        people_search_by_name = PeopleSearchByName(self._llm, show_logs=self._show_logs)

        # nodes
        planner_node = PlannerNode(self._llm, show_logs=self._show_logs)
        executor_node = ExecutorNode(self._llm, [people_search, people_search_by_name], show_logs=self._show_logs)

        # graph
        workflow.add_node("planner", planner_node.invoke)
        workflow.add_node("executor", executor_node.invoke)

        workflow.add_edge(START, "planner")
        workflow.add_edge("planner", "executor")
        workflow.add_edge("executor", END)

        return workflow.compile()

    def invoke(self, state: AgentState) -> AgentState:
        return AgentState.model_validate(dict(self._graph.invoke(state)))


if __name__ == "__main__":
    from langchain_core.messages import HumanMessage
    from app.agent.llms import LLMFactory

    state = AgentState(history=[HumanMessage("Есть ли жена у Киану Ривза?")], user_id="test_user")
    gpt = LLMFactory.get_llm("gpt-4o")

    agent = MovieAgent(gpt, show_logs=True)
    result = agent.invoke(state)
    print(result)
