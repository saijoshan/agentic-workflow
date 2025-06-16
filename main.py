from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from graph import build_graph
from langgraph.types import interrupt, Command

app = FastAPI()

# ✅ CORS middleware setup
app.add_middleware(
    CORSMiddleware, 
    allow_origins=["*"],  # Replace "*" with specific origin(s) in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

workflow_app = build_graph()

class InputPayload(BaseModel):
    query: str
    thread_id: str
    resume_flow: bool = False
    args: dict = {}

@app.post("/run-agent")
async def run_workflow(input: InputPayload):
    config = {"configurable": {"thread_id": input.thread_id}}  # Customize if needed    

    state = {
        "messages": [
            ("user", input.query),
        ],
        "task": input.query
    }

    all_messages = []
    if (input.resume_flow == True):
        if (input.args == {}):
            for event in workflow_app.stream(Command(resume=input.query),
                                    config,
                                    stream_mode="values"):
                if ('messages' in event and event['messages'][-1].content != ''):
                    all_messages.append({'message': event['messages'][-1].content})
                if ('__interrupt__' in event):  
                    all_messages.append(event['__interrupt__'][-1].value)
        else:
            print('>>> resuming flow')
            for event in workflow_app.stream(Command(resume=input.args),
                                    config,
                                    stream_mode="values"):
                if ('messages' in event and event['messages'][-1].content != ''):
                    all_messages.append({'message': event['messages'][-1].content})
                if ('__interrupt__' in event):  
                    all_messages.append(event['__interrupt__'][-1].value)
                
    else:
        for event in workflow_app.stream(state, config, stream_mode="values"):
            # print(event)
            # print('--------------------------------')
            if ('messages' in event and event['messages'][-1].content != ''):
                all_messages.append({'message': event['messages'][-1].content})
            if ('__interrupt__' in event):  
                all_messages.append(event['__interrupt__'][-1].value)
    return all_messages[-1]
