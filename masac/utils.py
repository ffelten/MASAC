def extract_agent_id(agent_str):
    """Extract agent id from agent string.

    Args:
        agent_str: Agent string in the format of "agent_{id}"

    Returns: (int) Agent id

    """
    return int(agent_str.split("_")[1])
