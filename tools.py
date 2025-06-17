from typing import Optional
import requests
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langgraph.types import interrupt
from langchain_core.messages import ToolMessage
import json

class BasicToolNode:
    """A node that runs the tools requested in the last AIMessage."""

    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, inputs: dict):
        if messages := inputs.get("messages", []):
            message = messages[-1]
        else:
            raise ValueError("No message found in input")
        outputs = []
        for tool_call in message.tool_calls:
            tool_result = self.tools_by_name[tool_call["name"]].invoke(
                tool_call["args"]
            )
            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
            break
        return {"messages": outputs}


class DataCollectionInputSchema(BaseModel):
    """Create a Data Collection in the dashboard"""
    name: str = Field(description="Name of the data collection")
    type: str = Field(description="Type of the data collection, either it can be 'General' or 'Face Recognition'")
    description: str = Field(description="Description of the collection")

@tool("create_data_collection", args_schema=DataCollectionInputSchema)
def create_data_collection(name: str, type: str, description: str):
    # Ask for confirmation from the user
    response = interrupt(
        {'interrupt': f"Trying to call `create_data_collection` with args {{'name': '{name}', 'type': '{type}', 'description': '{description}'}}. "
        "Please approve or suggest edits.", 'args': {'name': name, 'type': type, 'description': description, 'send': 'Yes/No'}}
    )

    if response["send"] == "Yes":
        try:    
            # API call to Next.js to insert a box (data collection)
            res = requests.post(
                "http://localhost:3000/api/boxes",  # ✅ Update URL if needed
                json={"name": response['name'], "type": response['type'], "description": response['description']},
                headers={"Content-Type": "application/json"},
                timeout=5
            )
            res.raise_for_status()
            data = res.json()
            return {
                "output": f"✅ Successfully created a data collection with name: {data['box']['name']}, "
                          f"type: {data['box']['type']}, description: {data['box']['description']}"
            }
        except Exception as e:
            return {"output": f"❌ Failed to create data collection: {str(e)}"}

    elif response["type"] == "No":
        return {"output": "user did not agree to create the collection"}

@tool("get_all_data_collections")
def get_all_data_collection() -> str:
    """Get brief details of all existing data collections from the dashboard"""
    try:
        response = requests.get("http://localhost:3000/api/boxes")
        response.raise_for_status()
        data = response.json()
        return data["boxes"]
    except Exception as e:
        return f"Failed to fetch data collections: {str(e)}"
    
class DataCollectionByName(BaseModel):
    """Get complete details of a specific collections including its id"""
    name: str = Field(description="Name of the data collection")
    
@tool("get_collection_by_name", args_schema=DataCollectionByName)
def get_collection_by_name(name: str) -> str:
    try:
        response = requests.get(
            "http://localhost:3000/api/boxes",
            params={"name": name}
        )
        response.raise_for_status()
        data = response.json()
        return data["box"]  # because your Next.js returns { box: {...} }
    except requests.HTTPError as e:
        if response.status_code == 404:
            return f"Data collection with name '{name}' not found."
        return f"HTTP error: {str(e)}"
    except Exception as e:
        return f"Failed to fetch data collection: {str(e)}"
    

class UpdateDataCollectionInputSchema(BaseModel):
    """Update a Data Collection in the dashboard, 'id' is mandatory to use this tool"""
    id: str = Field(description="id of the data collection")    
    name: Optional[str] = Field(default=None, description="same or new name for the data collection")
    description: Optional[str] = Field(default=None, description="same or new description for the data collection")
    type: Optional[str] = Field(default=None, description="same or new type for the data collection")

@tool("update_data_collection", args_schema=UpdateDataCollectionInputSchema)        
def update_data_collection(id: str, name: Optional[str] = None, description: Optional[str] = None, type: Optional[str] = None):

    update_args = {}
    if name is not None:
        update_args['name'] = name
    if type is not None:
        update_args['type'] = type
    if description is not None:
        update_args['description'] = description

    # Human-friendly message
    update_args_str = ', '.join(f"{k}: '{v}'" for k, v in update_args.items())

    response = interrupt(
        {
            'interrupt': f"Trying to update the collection with args {{{update_args_str}}}. "
                         "Please approve or suggest edits.",
            'args': {**update_args, 'send': 'Yes/No', 'feedback': ''}
        }
    )
    
    if response["send"] == "Yes":
        # Prepare update data
        update_data = {"id": id}
        if name:
            update_data["name"] = response['name']
        if description:
            update_data["description"] = response['description']
        if type:
            update_data["type"] = response['type']
        
        # Make API call
        api_url = "http://localhost:3000/api/boxes"  # Update with your actual API URL
        response = requests.put(api_url, json=update_data)
        
        if response.status_code == 200: 
            return {'message': 'Successfully updated'}
        else:
            return {'message': 'Backend api failed to update, please try later'}
    else:
        return {"message": f"User, rejected to update the collection, USER FEEDBACK: {response['feedback']}"}
    

class DeleteDataCollectionInputSchema(BaseModel):
    """Delete a Data Collection in the dashboard, 'id' is mandatory to use this tool"""
    id: str = Field(description="id of the data collection")

@tool("delete_data_collection", args_schema=DeleteDataCollectionInputSchema)
def delete_data_collection(id: str):

    # Ask user for confirmation
    response = interrupt(
        {
            'interrupt': f"Trying to delete the collection with id: {id}. Please approve or reject.",
            'args': {'send': 'Yes/No'}
        }
    )

    if response["send"] == "Yes":
        try:
            # Call the DELETE API with id as a query parameter
            api_url = "http://localhost:3000/api/boxes"
            res = requests.delete(api_url, params={"id": id})

            if res.status_code == 200:
                return {'message': 'Successfully deleted'}
            elif res.status_code == 404:
                return {'message': 'Collection not found'}
            else:
                return {'message': f'Failed to delete: {res.status_code} {res.text}'}
        except Exception as e:          
            return {'message': f'Error during deletion: {str(e)}'}
    else:
        return {"message": "User rejected the deletion request."}
    

class TalkToHuman(BaseModel):
    """Talk to user or ask questions or provide any insights to the user"""

    question: str = Field(description="A question to the user.")


@tool("talk_to_human", args_schema=TalkToHuman)
def talk_to_human(question: str):
    response = interrupt({'interrupt': question, 'args': {}})
    # if response['proceed'] == 'Yes' and response['inputs'] != '':
    #     return {"message": f"user inputs: {response['inputs']}"} 
    # elif response['proceed'] == 'Yes' and response['inputs'] == '':
    #     return {"message": f"thanks, now proceed"} 
    # else:
    #     return {"message": "User requested to end the complete execution"}  
    return response