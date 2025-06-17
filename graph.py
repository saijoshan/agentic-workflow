from langgraph.prebuilt import ToolNode
from langgraph.graph import MessagesState, START, END, StateGraph
from langgraph.checkpoint.redis import RedisSaver
from IPython.display import Image, display
from langchain_core.messages import ToolMessage
from langgraph.types import interrupt, Command
from tools import create_data_collection, get_all_data_collection, get_collection_by_name, update_data_collection, delete_data_collection, talk_to_human, BasicToolNode
from utility import filter_history
from agents import get_main_agent, get_planner, get_replanner, Response
import json

class DataCollectionState(MessagesState):
    task: str
    plan: list
    current_instruction: str

create_data_collection_tool = BasicToolNode([create_data_collection])
get_all_data_collection_tool = BasicToolNode([get_all_data_collection])
get_collection_by_name_tool = BasicToolNode([get_collection_by_name])
update_data_collection_tool = BasicToolNode([update_data_collection])
delete_data_collection_tool = BasicToolNode([delete_data_collection])
talk_to_human_tool = BasicToolNode([talk_to_human])

def get_memory():
    REDIS_URI = "redis://localhost:6379/0"
    memory = None
    with RedisSaver.from_conn_string(REDIS_URI) as cp:
        cp.setup()
    memory = cp
    return memory

def run_planner(state):
    planner = get_planner()
    plan = planner.invoke({"messages": [("user", state["task"])]})          
    plan_str = '\n'.join(plan.steps)
    print(f"main plan >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> {plan_str}")      
    return {"plan": plan.steps, 'current_instruction': plan.steps[0]}

def run_replanner(state):
    message_Str, message_list = filter_history(state['messages'])
    replanner = get_replanner(state['task'], state['plan'], message_Str)
    return replanner

def run_agent(state):           
    message_Str, message_list = filter_history(state['messages'])
    return get_main_agent(state['current_instruction'], message_Str)  

def should_continue(state):
    messages = state["messages"]
    last_message = messages[-1]

    if not last_message.tool_calls:
        return 'replanner'
    elif last_message.tool_calls[0]["name"] == "create_data_collection":
        return "create_data_collection_tool"
    elif last_message.tool_calls[0]["name"] == "get_all_data_collections":
        return "get_all_data_collection_tool"
    elif last_message.tool_calls[0]["name"] == "get_collection_by_name":
        return "get_collection_by_name_tool"
    elif last_message.tool_calls[0]["name"] == "update_data_collection":
        return "update_data_collection_tool"
    elif last_message.tool_calls[0]["name"] == "delete_data_collection":
        return "delete_data_collection_tool"
    elif last_message.tool_calls[0]["name"] == "talk_to_human":
        return "talk_to_human_tool"
    
def should_end(state: DataCollectionState):
    if state['current_instruction'] == 'END':
        return END
    else:
        return "agent"

def build_graph():
    workflow = StateGraph(DataCollectionState)

    workflow.add_node("planner", run_planner)
    workflow.add_node("agent", run_agent)
    workflow.add_node("create_data_collection_tool", create_data_collection_tool)
    workflow.add_node("get_all_data_collection_tool", get_all_data_collection_tool)
    workflow.add_node("get_collection_by_name_tool", get_collection_by_name_tool)
    workflow.add_node("update_data_collection_tool", update_data_collection_tool)
    workflow.add_node("delete_data_collection_tool", delete_data_collection_tool)
    workflow.add_node("talk_to_human_tool", talk_to_human_tool)
    workflow.add_node("replanner", run_replanner)

    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "agent")
    workflow.add_conditional_edges("agent", should_continue, path_map=["create_data_collection_tool", "get_all_data_collection_tool", "get_collection_by_name_tool", "update_data_collection_tool", "delete_data_collection_tool", "talk_to_human_tool", "replanner"])
    workflow.add_edge("create_data_collection_tool", "replanner")
    workflow.add_edge("get_all_data_collection_tool", "replanner")
    workflow.add_edge("get_collection_by_name_tool", "replanner")
    workflow.add_edge("update_data_collection_tool", "replanner")
    workflow.add_edge("delete_data_collection_tool", "replanner")
    workflow.add_edge("talk_to_human_tool", "replanner")
    workflow.add_conditional_edges("replanner", should_end, ["agent", END])
                                   

    graph = workflow.compile(checkpointer=get_memory())

    # from pathlib import Path
    # display(Image(graph.get_graph().draw_mermaid_png()))

    # img_data = graph.get_graph().draw_mermaid_png()
    # output_path = Path("graph.png")
    # with open(output_path, "wb") as f:
    #     f.write(img_data)
    return graph
