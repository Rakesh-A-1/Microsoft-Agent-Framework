import json
import os
from datetime import datetime
from agent_framework import AgentThread

FILE_PATH = "thread_history.json"

async def save_thread(thread: AgentThread):
    """Serialize and append the thread state to storage (never overwrite history)."""
    try:
        serialized = await thread.serialize()
    except Exception as e:
        print(f"Could not serialize thread: {e}")
        return

    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "thread_state": serialized
    }

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

async def restore_last_thread(agent) -> AgentThread:
    """
    Restore the most recently saved AgentThread, if exists.
    If no storage or error, returns a fresh thread (if supported).
    """
    # Load history
    if not os.path.exists(FILE_PATH):
        print("No stored session found. Starting fresh.")
        if hasattr(agent, "get_new_thread"):
            return agent.get_new_thread()
        else:
            raise RuntimeError("Agent does not support get_new_thread() — cannot create a new thread.")

    try:
        with open(FILE_PATH, "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading thread history: {e}. Starting fresh.")
        if hasattr(agent, "get_new_thread"):
            return agent.get_new_thread()
        else:
            raise RuntimeError("Agent does not support get_new_thread() — cannot create a new thread.")

    if not data:
        print("Thread history empty. Starting fresh.")
        if hasattr(agent, "get_new_thread"):
            return agent.get_new_thread()
        else:
            raise RuntimeError("Agent does not support get_new_thread() — cannot create a new thread.")

    latest = data[-1].get("thread_state")
    if latest is None:
        print("No valid thread_state in last entry. Starting fresh.")
        if hasattr(agent, "get_new_thread"):
            return agent.get_new_thread()
        else:
            raise RuntimeError("Agent does not support get_new_thread() — cannot create a new thread.")

    # Try to deserialize if agent supports it
    if hasattr(agent, "deserialize_thread"):
        try:
            thread = await agent.deserialize_thread(latest)
            print("Restored saved conversation thread.")
            return thread
        except Exception as e:
            print(f"Deserialization failed: {e}. Starting fresh.")
            if hasattr(agent, "get_new_thread"):
                return agent.get_new_thread()
            else:
                raise RuntimeError("Agent does not support get_new_thread() — cannot create a new thread.")
    else:
        raise RuntimeError("Agent does not support deserialize_thread() — cannot restore thread.")