import os
from dotenv import load_dotenv
import streamlit as st

from langgraph.graph import MessagesState, StateGraph, END, START
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.prebuilt import tools_condition, ToolNode
from langchain_openai import AzureChatOpenAI
from langchain_core.tools import tool

load_dotenv()

# Set up LLM with Azure OpenAI
llm = AzureChatOpenAI(
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),  
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version=os.getenv("AZURE_OPENAI_VERSION"),               
    temperature=0.7,
    model_name="gpt-4o"
)

# Define tools with proper type annotations
@tool
def add(a: int, b: int) -> int:
    """Addition of two numbers"""
    return a + b

@tool
def multiply(a: int, b: int) -> int:
    """Multiplication of two numbers"""
    return a * b

@tool
def divide(a: float, b: float) -> float:
    """Division of two numbers (with zero division check)"""
    if b == 0:
        raise ValueError("Division by zero is not allowed.")
    return a / b

# DuckDuckGo search tool
search = DuckDuckGoSearchRun()

# List of all tools
tools = [add, multiply, divide, search]
llm_with_tools = llm.bind_tools(tools)

sys_msg = SystemMessage(content="You are a helpful Assistant that can do math and fetch real-world info when needed.")

# Reasoner node
def reasoner(state: MessagesState):
    return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}

# Build the LangGraph
builder = StateGraph(MessagesState)
builder.add_node("reasoner", reasoner)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "reasoner")
builder.add_conditional_edges("reasoner", tools_condition)
builder.add_edge("tools", "reasoner")
react_graph = builder.compile()

# Streamlit UI
st.set_page_config(page_title="LangGraph Assistant", page_icon="🤖")
st.title("LangGraph + Azure GPT-4o 🤖")

user_input = st.text_input("Ask a question or give a math query:", key="input")

if user_input:
    with st.spinner("Processing..."):
        message = [HumanMessage(content=user_input)]
        result = react_graph.invoke({"messages": message})
        st.success("Response:")
        for msg in result['messages']:
            st.markdown(f"**{msg.type.capitalize()}:** {msg.content}")
