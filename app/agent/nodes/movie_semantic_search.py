from pathlib import Path
from dotenv import load_dotenv

from app.core.index_db import collection
from app.agent.nodes.planner_node import MOVIES_SEARCH_FIELDS as OUTPUT_FIELDS
from app.agent.nodes._base_api_tool import BaseApiTool


load_dotenv(Path(__file__).parent.parent.parent.parent.resolve() / ".env")


class MovieSemanticSearch(BaseApiTool):

    def __init__(
        self,
        name = "MovieSemanticSearch",
        description = "Осуществляет семантический поиск по содержанию фильмов и возвращает информацию о них",
        distance_thr = 0.9,
        show_logs: bool = False,
    ):  
        self._name = name
        self._description = description
        self._distance_thr = distance_thr
        self._show_logs = show_logs


    def _invoke(self, question: str, collected_info: str, *args, **kwargs) -> str:

        if self._show_logs:
            print(f"---{self._name}---")
            print(question)

        response = collection.query(query_texts=question, n_results=1)
        if response['distances'][0][0] > self._distance_thr:
            answer = "К сожалению, я не могу найти информацию по вашему запросу"
        else:
            answer = response['metadatas'][0][0]['movie_data']
        if self._show_logs:
            print(answer)
            print("-------------------")

        return answer


if __name__ == "__main__":
    from app.agent.llms import LLMFactory

    gpt = LLMFactory.get_llm("deepinfra/Llama-3.3-70B-Instruct")

    search = MovieSemanticSearch(gpt, show_logs=True)

    print(search.invoke("Фильм про человека, осужденного на смертную казнь за совершённое другим убийство, и обладающего даром исцеления", ""))
