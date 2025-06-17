def filter_history(messages):
    result = []
    
    for message in messages:
        if hasattr(message, 'content') or isinstance(message, dict):
            # Handle dict format (from your JSON)
            if isinstance(message, dict):
                msg_type = message.get('type', '')
                content = message.get('content', '')
                tool_calls = message.get('tool_calls', [])
            else:
                # Handle object format (LangChain message objects)
                msg_type = type(message).__name__
                content = getattr(message, 'content', '')
                tool_calls = getattr(message, 'tool_calls', [])
            
            if 'Human' in msg_type:
                result.append(f"HUMAN: {content}")
            elif 'AI' in msg_type:
                if tool_calls:
                    for tool_call in tool_calls:
                        tool_name = tool_call.get('name', '')
                        tool_args = tool_call.get('args', {})
                        result.append(f"AI: called tool -> {tool_name} with args {tool_args}")
                elif content:
                    result.append(f"AI: {content}")
            elif 'Tool' in msg_type:
                tool_name = message.get('name', 'unknown_tool') if isinstance(message, dict) else getattr(message, 'name', 'unknown_tool')
                result.append(f"TOOL: {tool_name} returned -> {content}")
    return '\n'.join(result), result