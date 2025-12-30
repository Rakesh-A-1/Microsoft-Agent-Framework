import json
import os
from datetime import datetime
from agent_framework import AgentThread

FILE_PATH = "thread_history.json"

async def save_thread(thread: AgentThread, text_history: list):
    """
    Serialize and append the thread state AND text history to storage.
    """
    try:
        serialized = await thread.serialize()
    except Exception as e:
        print(f"Could not serialize thread: {e}")
        return

    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "thread_state": serialized,
        "text_history": text_history  # Saving the UI chat history here
    }

    # Load existing data to append, or start fresh
    if os.path.exists(FILE_PATH):
        try:
            with open(FILE_PATH, "r") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            data = []
    else:
        data = []

    data.append(entry)

    with open(FILE_PATH, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Conversation state saved to {FILE_PATH}")

async def restore_last_thread(agent):
    """
    Restore the most recently saved AgentThread and the last 5 UI messages.
    Returns: (thread, recent_history_list)
    """
    # 1. Check if file exists
    if not os.path.exists(FILE_PATH):
        print("No stored session found. Starting fresh.")
        if hasattr(agent, "get_new_thread"):
            return await agent.get_new_thread(), []
        else:
            raise RuntimeError("Agent does not support get_new_thread()")

    try:
        with open(FILE_PATH, "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading thread history: {e}. Starting fresh.")
        if hasattr(agent, "get_new_thread"):
            return agent.get_new_thread(), []
        else:
            raise RuntimeError("Agent does not support get_new_thread()")

    if not data:
        print("Thread history empty. Starting fresh.")
        if hasattr(agent, "get_new_thread"):
            return await agent.get_new_thread(), []
        else:
            raise RuntimeError("Agent does not support get_new_thread()")

    # Get the last entry
    last_entry = data[-1]
    latest_state = last_entry.get("thread_state")

    if latest_state is None:
        print("No valid thread_state in last entry. Starting fresh.")
        if hasattr(agent, "get_new_thread"):
            # Return empty list for history if state is invalid
            return await agent.get_new_thread(), []
        else:
            raise RuntimeError("Agent does not support get_new_thread()")

    # 2. Restore the Agent Thread (Logic)
    thread = None
    if hasattr(agent, "deserialize_thread"):
        try:
            thread = await agent.deserialize_thread(latest_state)
            print("Restored saved conversation thread.")
        except Exception as e:
            print(f"Deserialization failed: {e}. Starting fresh.")
            if hasattr(agent, "get_new_thread"):
                thread = await agent.get_new_thread()
            else:
                raise RuntimeError("Agent does not support get_new_thread()")
    else:
        raise RuntimeError("Agent does not support deserialize_thread()")

    # 3. Restore and Slice UI History (Logic for "Last 5 Only")
    full_history = last_entry.get("text_history", [])
    
    # Slice: take the last 5 items. If list is shorter than 5, take everything.
    recent_history = full_history[-5:] if len(full_history) > 5 else full_history
    
    return thread, recent_history