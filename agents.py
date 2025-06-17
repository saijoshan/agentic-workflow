import os
from typing import List, Union
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from tools import create_data_collection, get_all_data_collection, get_collection_by_name, update_data_collection, delete_data_collection, talk_to_human
load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY is not set. Please export it or add it to .env") 

class Response(BaseModel):
        """Response to user."""

        response: str

class Plan(BaseModel):
    """Plan to follow in future"""
    steps: List[str] = Field(
        description="List of concrete, ordered steps to follow. Each step must use only one tool."
    )

class Act(BaseModel):
    """Updated execution plan"""

    action: Union[Response, Plan] = Field(
        description="Action to perform. If you want to respond to user, use Response. "
        "If you need to further use tools to get the answer, use Plan."
    )


def get_main_agent(query, metadata):

    model = init_chat_model("openai:gpt-4.1")

    tools = [create_data_collection, get_all_data_collection, get_collection_by_name, update_data_collection, delete_data_collection, talk_to_human]

    model = model.bind_tools(tools)    

    prompt = f"""
    You are an expert data_collection manager inside a dashboard, you are equipped with necessary tools to complete given tasks
    Your main goal is to complete the current task.

    CURRENT TASK
    {query}

    METADATA
    {metadata}
    
    INSTRUCTIONS
    - Just perform the tasks which are given to you(Do not do additional tasks on your own).
    - Use the information mentioned in the metadata if you want.
    - You must use only one tool at once.
    """
    print(f"agent running with >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> {prompt}")
    response = model.invoke(prompt)
    return {"messages": [response]}

def get_planner():
    class Plan(BaseModel):
        """A clear, ordered list of executable steps for the agent."""
        steps: List[str] = Field(
            description="Each step must clearly state ONE action using exactly ONE tool. Steps must be executable in order without ambiguity."
        )

    planner_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are a **Precise Planner** for a dashboard automation system with multiple data collections in it.

Your role:  
Given any user request about managing data collections, break it down into an ordered list of **atomic steps** for an agent to execute.

**Each step must do only ONE action using exactly ONE tool.**  
Steps must be ordered logically. If any needed information is missing, **insert a step to ask the user before continuing.**

---

## **Available Tools**

- **create_data_collection**: Create a new data collection. Needs: `name`, `type` ('General' or 'Face Recognition'), `description`.
- **get_all_data_collections**: Retrieve a list of all existing collections.
- **get_collection_by_name**: Get details along with ID of a specific collection by its name.
- **update_data_collection**: Update a collection. Needs: `id` plus any of `name`, `type`, `description`.
- **delete_data_collection**: Delete a collection by `id`.
- **talk_to_human**: To explain user anything, provide user insights or ask the user any question to get missing information.   

---

## **Guidelines**

1 Use exactly ONE tool per step — no multi-tool actions.  
2 Always retrieve collection IDs before updating or deleting.  
3 If you don't have a required value (like a new description), insert a **talk_to_human** step immediately before the action.
4 To explain anything to user or provide any collection info to user, add **talk_to_human** step. 
5 Steps must be simple, direct instructions to the agent:           
   Example:  
   - "Get details of collection named 'My Collection' using get_collection_by_name"
   - "Ask the user: What should be the new description for collection ID 123?"
   - "Update collection with ID 123 using update_data_collection with the new description."

6 Do not invent tools or actions that are not listed.   

---

## **Output Format**

Return only the plan —  
A numbered list of clear, plain-text steps, each describing which tool will be used and for what.

Do not add extra explanation, apologies, or commentary.
""",
            ),
            ("placeholder", "{messages}"),
        ]
    )

    planner = planner_prompt | ChatOpenAI(
        model="gpt-4.1",
        temperature=0
    ).with_structured_output(Plan)

    return planner

def get_replanner(task, plan, history):

    replanner_prompt = f"""
You are a **Precise instructor** for an agentic system that executes one step at a time.

## Objective:
{task}

## Original Plan:
{plan}

## Previous Step Results:
{history}

---

## **Your job:**

1. Carefully Inspect the results of completed steps and identify the state of the task.
2. Create a proper instruction for an agent with tool name and its parameters in a sentence.

## **Agent has below tools:**
- **create_data_collection**: Create a new data collection. Needs: `name`, `type` ('General' or 'Face Recognition'), `description`.
- **get_all_data_collections**: Retrieve a list of all existing collections.
- **get_collection_by_name**: Get details and ID of a specific collection by its name.
- **update_data_collection**: Update a collection. Needs: `id` plus any of `name`, `type`, `description`.
- **delete_data_collection**: Delete a collection by `id`.
- **talk_to_human**: Provide user insights of any tool output or ask the user any question to get missing information or confirmation.


---

## **Output Format**
- A clear single step instruction containing all the information.

## IMPORTANT NOTE
- Once all the steps are completed and the objective is achieved just return 'END' without any special characters.
"""

    llm = init_chat_model("openai:gpt-4.1")
    response = llm.invoke(replanner_prompt)

    return {'current_instruction': response.content}            