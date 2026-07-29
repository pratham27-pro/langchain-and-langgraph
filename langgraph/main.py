import os
from dotenv import load_dotenv
from langcain_core.messages import HumanMessage
from langgraph.graph import MessageState, StateGraph

from nodes import run_agent_reasoning, tool_node

load_dotenv()

AGENT_REASON = "agent_reason"
AGENT_ACT = "act"
LAST = -1

 