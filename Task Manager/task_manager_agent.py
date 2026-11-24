#===============================Necessary Imports=================================#
from langgraph.graph import StateGraph, END
from langchain_core.messages import AIMessage, ToolMessage, SystemMessage, HumanMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.graph.message import add_messages
from typing import TypedDict, Sequence, List, Annotated
from dotenv import load_dotenv
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition
import json
from pathlib import Path

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR/"agent_data.json"

files = {}
active_file = None

#=================Functions to support tool Operation=============================#

def save_agent_data():
    data = {
        "files" : files,
        "active_file" : active_file
    }
    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=4)

def load_agent_data():
    global files, active_file

    if DATA_FILE.exists():
        with open(DATA_FILE, "r") as f:
            data = json.load(f)
        files = data.get("files", {})
        active_file = data.get("active_file")
        print("Existing file loaded successfully!")
    else:
        files = {}
        active_file = None
        print("No Existing files, Please make one!")

load_agent_data()

def update_saved_file():
    if active_file is None:
        return
    tasks = files.get(active_file)
    
    if tasks is None:
        return

    file_path = BASE_DIR / active_file
    
    with open(file_path, "w") as f:
        for task in tasks:
            f.write(f"Task : {task['task']}\n")
            f.write(f"Status : {task['status']}\n")
            f.write("-" * 30 + "\n")

def get_active_files():
    if active_file is None:
        return None

    return files.get(active_file)
            
class AgentState(TypedDict):
    messages : Annotated[Sequence[BaseMessage], add_messages]

#====================================Tools======================================#

@tool
def create_file(filename: str) -> str:
    """Create a new empty To-Do List and make it the active list."""
    global active_file

    if filename in files:
        return f"'{filename}' already exists. Please select that list instead."

    files[filename] = []
    active_file = filename

    save_agent_data()
    update_saved_file()

    return f"'{filename}' has been created successfully and is now the active list."
    

@tool
def add_task(content : str) -> str:
    """ This tool adds the task description and status in the task list"""
    tasks = get_active_files()
    
    if tasks is None:
        return "Please create or select the tasks file first!"
    
    tasks.append(
        {
            "task" : content,
            "status" : "Pending"
        }
    )
    save_agent_data()
    update_saved_file()
    return f"Task {content} is added to To-Do-List!"

@tool
def complete_task(content : str)->str:
    """ This tool marks the completed task as completed based on the description and gives appropriate warning message if task is not present in the existing To-Do-List"""
    
    tasks = get_active_files()
    for task in tasks:
        if task["task"].lower() == content.lower():
            task["status"] = "Completed"
            save_agent_data()
            update_saved_file()
            return f"The status of task {content} is updated to {task['status']}"
    return f"Sorry! To-Do-List does not have the provided task {content}"

@tool
def delete_task(content : str)->str:
    """ This tool deletes content of the To-Do-List as per user reuirement"""
    tasks = get_active_files()
    for task in tasks:
        if task["task"].lower() == content.lower():
            tasks.remove(task)
            save_agent_data()
            update_saved_file()
            return f"Task {content} has been removes successfully!"
    return f"Task {content} was not found!"
    
@tool
def save_list(filename : str)->str:
    """ This tool saves the all tasks in a text file and keeps track of any updation or modification in the task"""
    global active_file
    
    if active_file is None:
        return "No Active files to save!"
    
    if filename in files:
        return f"{filename} already exists. Choose another filename."
    
    files[filename] = files[active_file].copy
    
    active_file = filename
    save_agent_data()
    update_saved_file
    return f"file {filename} has been saved successfully!"

@tool
def delete_list(filename : str)->str:
    """Delete an entire task list, including its JSON entry and physical text file."""
     
    file_path = BASE_DIR/filename
    global active_file
    
    if filename not in files:
        return f"File '{filename}' does not exists!"
        
    
    if file_path.exists():
        file_path.unlink()
        
    del files[filename]
    
    if active_file.lower() == filename.lower():
        active_file = None
    save_agent_data()
    
    return f"File '{filename}' Has been deleted successfully!"

@tool
def switch_file(filename : str)->str:
    """Switch the active task list to an existing file."""
    global active_file
    
    matched_file = next((name for name in files if name.lower() == filename.lower()), None)
    if matched_file is None:
        return f"File '{filename}' does not exists! existing files : {list(files.keys())}"
    active_file = matched_file
    save_agent_data
    return f"Successfully switched to '{active_file}'. It is now the active list."

@tool
def list_files()-> str:
    """ Display all the to do lists and identify the active one"""
    if not files:
        return "No existing To-Do-Lists!"
    
    result = ""
    
    for filename,tasks in files.items():
        if filename.lower() == active_file.lower():
            result += f"- {filename} (Active)\n"
        else:
            result += f"- {filename}\n"
        
        if not tasks:
            result += "No tasks available!\n"
            continue
        for i, task in enumerate(tasks, start=1):
            result += f"{i}. {task['task']} - {task['status']}\n"
            
    return result
    

tools = [create_file, add_task, complete_task, delete_task, save_list, delete_list, switch_file, list_files]

llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0, max_tokens = 500).bind_tools(tools)

#=====================================Agent=======================================#

def our_agent(state : AgentState) -> AgentState:
    sys_prompt = SystemMessage(content= f""" 
You are a helpful Task Manager Agent that manages multiple To-Do Lists.

Available tools: list_files, create_file, switch_file, add_task, complete_task, delete_task, save_list, delete_list.

Rules:
1. Use list_files to view existing lists and their tasks.
2. Restore the previously active list when the application starts.
3. Keep the active list unchanged when viewing other lists. Switch only when explicitly requested.
4. Perform all task modifications on the active list. Use switch_file before modifying another existing list.
5. Use create_file for new empty lists and save_list only to copy an existing list.
6. Never delete an entire list when asked to delete an individual task.
7. Confirm operations only after tool success and display the updated list after modifications.
                               """ )
    
    if not state["messages"]:
        user_input = "I am ready to help with the To-Do-List Creation what would you like to do?"
        user_message = HumanMessage(content=user_input)
    else:
        user_input = input("What's the next task?--")
        print(f"\nUSER : {user_input}")
        user_message = HumanMessage(content=user_input)
    all_mess = [sys_prompt]
    all_mess.extend(state["messages"])
    all_mess.append(user_message)
    
    response = llm.invoke(all_mess)
    
    print(f"AI Response : {response.content}")
    if hasattr(response, "tool_calls") and response.tool_calls:
        print(f"Using Tools : {[tc['name'] for tc in response.tool_calls]}")
    return{"messages" : [user_message, response]}


def print_mess(messages):
    if not messages:
        return
    for message in messages[-3:]:
        if isinstance(messages, ToolMessage):
            print(f"Tool Result: {ToolMessage}")

#======================================Graph=========================================#

graph = StateGraph(AgentState)
graph.add_node("agent", our_agent)
tools_node = ToolNode(tools)
graph.add_node("tools", tools_node)
graph.add_edge("agent", "tools")
graph.set_entry_point("agent")
graph.add_conditional_edges(
    "agent",
    tools_condition
)
graph.add_edge("tools", "agent")
app = graph.compile()

png_data = app.get_graph().draw_mermaid_png()
with open("Task_manager.png", "wb") as f:
    f.write(png_data)

def run_task_manager_agent():
    print("-" * 10 + "\n")
    state = {"messages" : []}    
    for step in app.stream(state, stream_mode="values"):
        if "messages" in step:
            print_mess(step["messages"])
    print("-" * 5 + " END " + "-" * 5 + "\n")
    
if __name__ == "__main__":
    run_task_manager_agent()


