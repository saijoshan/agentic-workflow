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
    """Action to perform."""

    action: Union[Response, Plan] = Field(
        description="Action to perform. If you want to respond to user, use Response. "
        "If you need to further use tools to get the answer, use Plan."
    )


def get_main_agent(query):

    model = init_chat_model("openai:gpt-4.1")

    tools = [create_data_collection, get_all_data_collection, get_collection_by_name, update_data_collection, delete_data_collection, talk_to_human]

    model = model.bind_tools(tools)    

    prompt = f"""
    You are an expert data_collection manager inside a dashboard, you are equipped with necessary tools to complete user tasks.
    Your goal is to complete user tasks and help the user with his queries. ALl the queries related to updating a collection forward it to update_collection_agent
    without asking any further questions.

    CURRENT USER QUERY
    {query}
    
    INSTRUCTIONS
    - Think carefully based on the history provided and act accordingly (Do not do additional tasks on your own).
    - You have an update_collection_agent under you, just forwards the task related to updating collections to update_collection_agent without asking questions.
    """

    response = model.invoke(prompt)
    return {"messages": [response]}

def get_planner():
    class Plan(BaseModel):
        """Plan to follow in future"""
        steps: List[str] = Field(
            description="List of concrete, ordered steps to follow. Each step must use only one tool."
        )

    planner_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a **Task Planner**, you are a part of a dashboard where user can create, update or delete data collections. Your job is to break down a user query into a clear, step-by-step execution plan for a multi-agent system automation.

## Available Tools:
- **create_data_collection**: Create a collection (params: `name`, `type` ['General'|'Face Recognition'], `description`).
- **get_all_data_collections**: Retrieve all existing collections.
- **ask_user**: Ask the user for any missing parameter or clarification.
- **update_collection_agent**: Update a collection (params: `exact_name`, `instruction`).

## Guidelines:
1. Use **exactly one tool per step**.
2. Steps must be ordered logically: retrieve data before updating or creating.
3. If any required information is missing, insert an **ask_user** step **immediately** before continuing.
4. Use clear, executable instructions in each step.
5. Only include relevant steps; do not hallucinate tools or actions.

## Output Format:
Return the plan as a numbered list of plain text steps.  
Each step must clearly name the tool and what it will do.

---

Generate only the steps. Do not add extra commentary.
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

def get_replanner():

    replanner_prompt = ChatPromptTemplate.from_template(
        """For the given objective, come up with a simple step by step plan. \
    This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. \
    The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.

    Your objective was this:
    {task}

    Your original plan was this:
    {plan}

    You have currently done the follow steps:
    {history}

    Update your plan accordingly. If no more steps are needed and you can return to the user, then respond with that. Otherwise, fill out the plan. Only add steps to the plan that still NEED to be done. Do not return previously done steps as part of the plan."""
    )


    replanner = replanner_prompt | ChatOpenAI(
        model="gpt-4.1", temperature=0
    ).with_structured_output(Act)

    return replanner