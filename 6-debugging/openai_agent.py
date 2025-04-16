from typing import Annotated
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langgraph.graph import START, END, MessageGraph
from langgraph.graph.state import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
import os
from dotenv import load_dotenv

load_dotenv()

os.environ["OPEN_API_KEY"] = os.getenv("OPEN_API_KEY")

os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")


# Define State
class state(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

model = ChatOpenAI(temperature=0)

def make_default_graph():
    """Make a simple LLM Agent"""
    graph_workflow = StateGraph(state)
    def call_model(state):
        return {"messages":[model.invoke(state["messages"])]}

    graph_workflow.add_node("agent", call_model)
    graph_workflow.add_edge("agent", END)
    graph_workflow.add_edge(START, "agent")

    agent = graph_workflow.compile()
    return agent

def make_alternative_graph():
    """Tool calling agent"""
    @tool
    def add(a: float, b: float):
        """add two numbers"""
        return a + b

    tool_node = ToolNode([add])
    model_with_tools = model.bind_tools(tool_node)

    def call_model(state):
        return {"messages":[model_with_tools.invoke(state["messages"])]}

    def should_continue(state: state):
        if state["messages"][-1].tool_calls:
            return "tools"
        else:
            return END

    graph_workflow = StateGraph(state)

    graph_workflow.add_node("agent", call_model)
    graph_workflow.add_node("tools", tool_node)
    graph_workflow.add_edge("tools", "agent")
    graph_workflow.add_edge(START, "agent")
    graph_workflow.add_conditional_edges("agent", should_continue)

    agent = graph_workflow.compile()
    return agent

agent = make_alternative_graph()
