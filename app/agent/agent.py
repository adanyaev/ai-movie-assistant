from typing import Annotated, Literal, TypedDict
import os
from datetime import datetime

from psycopg import Connection
from langchain_core.messages import HumanMessage, trim_messages
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langgraph.checkpoint.postgres import PostgresSaver

from app.core.config import settings
from . import tools as t


# llm = ChatGroq(
#     model="llama-3.1-70b-versatile",
#     #model="llama3-groq-70b-8192-tool-use-preview",
#     temperature=0,
#     max_tokens=None,
#     timeout=None,
#     max_retries=2,
# )

# llm = ChatOpenAI(
#                 model="gpt-4o",
#                 api_key=os.environ["OPENAI_API_KEY"],
#                 #temperature=0,
#                 max_tokens=None,
#                 timeout=None,
#                 max_retries=2,
#             )

llm = ChatOpenAI(
            #model="meta-llama/Meta-Llama-3.1-405B-Instruct",
            model="meta-llama/Llama-3.3-70B-Instruct",
            api_key=os.environ["DEEPINFRA_KEY"],
            base_url="https://api.deepinfra.com/v1/openai",
            # temperature=0,
            )


tools = [t.search_movies_with_filters, t.get_movie_by_title]

tool_node = ToolNode(tools)

primary_assistant_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful movie assistant. "
            " Use the provided tools to serve user's requests. "
            " Do not use your memory to answer questions, use only information received from tools calls."
            " When searching, be persistent."
            #" You must answer in russian."
            #"\n\nCurrent user:\n<User>\n{user_info}\n</User>"
            "\nCurrent time: {time}.",
        ),
        ("placeholder", "{messages}"),
    ]
).partial(time=datetime.now())

llm = primary_assistant_prompt | llm.bind_tools(tools)


trimmer = trim_messages(
    strategy="last",
    token_counter=len,
    max_tokens=5,
    start_on="human",
    end_on=("human", "tool"),
    include_system=False,
)


def should_continue(state: MessagesState) -> Literal["tools", END]:
    messages = state['messages']
    last_message = messages[-1]
    # If the LLM makes a tool call, then we route to the "tools" node
    if last_message.tool_calls:
        return "tools"
    # Otherwise, we stop (reply to the user)
    return END


def call_model(state: MessagesState):
    messages = state['messages']
    response = llm.invoke({'messages': trimmer.invoke(messages)})
    #response = llm.invoke({'messages': messages})
    return {"messages": [response]}



workflow = StateGraph(MessagesState)


workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)


workflow.add_edge(START, "agent")


workflow.add_conditional_edges(
    "agent",
    should_continue,
)

workflow.add_edge("tools", 'agent')

# postgres_pass = os.environ.get("POSTGRES_PASS")
# postgres_host = os.environ.get("POSTGRES_HOST")
# postgres_port = os.environ.get("POSTGRES_PORT")

DB_URI = settings.DB_URI

connection_kwargs = {
    "autocommit": True,
    "prepare_threshold": 0,
}

with Connection.connect(DB_URI, **connection_kwargs) as conn:
    checkpointer = PostgresSaver(conn)
    checkpointer.setup()
