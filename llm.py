from langchain_core.messages import BaseMessage
from typing import TypedDict, Sequence, Annotated
from operator import add as add_messages
from config import GROQ_API_KEY
from groq import Groq
from tools import ragQueryTool
import json


# model = init_chat_model("google_genai:gemini-2.5-flash-lite", 
#                         api_key=GEMINI_API_KEY,
#                         temperature=0.2,
#                         max_output_tokens=512,
#                         top_p=0.8
#                         )

# llm = ChatGroq(
#     api_key=GROQ_API_KEY,
#     model = "groq/compound",
#     temperature=0.2,
#     max_tokens=512,
#     timeout=None,
# )

client = Groq(api_key=GROQ_API_KEY)

toolsSchema = [{
    "type": "function",
    "function": {
        "name": "ragQueryTool",
        "description": "Perform a RAG query on the dense index.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The query to perform the RAG search with."
                }
            },
            "required": ["query"]
        }
    }
}]

available_tools = {
    "ragQueryTool": ragQueryTool
}

def execute_tool_call(tool_call):
    """Parse and execute a single tool call"""
    function_name = tool_call.function.name
    function_to_call = available_tools[function_name]
    function_args = json.loads(tool_call.function.arguments)
    
    # Call the function with unpacked arguments
    return function_to_call(**function_args)

messages = [
    {
        "role": "system",
        "content": "You are a helpful assistant for answering questions about the content of a PDF document that has been uploaded. You have access to a tool called 'ragQueryTool' that allows you to perform RAG queries on the document's content. Use this tool to find relevant information in the document to answer the user's questions."
    }
]

def generate_response(user_query):
    try:
        messages.append({"role": "user", "content": user_query})
        
        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=messages,
            tools=toolsSchema,
            temperature=0.2,
            max_tokens=512,
            timeout=None,
        )
        
        messages.append(response.choices[0].message)
        
        if response.choices[0].message.tool_calls:
            for tool_call in response.choices[0].message.tool_calls:
                tool_result = execute_tool_call(tool_call)
                
                messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": tool_call.function.name,
                "content": str(tool_result)
            })
                
            final_response = client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=messages,
                tools=toolsSchema,
                temperature=0.2,
                max_tokens=512,
                timeout=None,
            )
            
            return final_response.choices[0].message.content
            
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error generating response: {e}")
        return "Sorry, I encountered an error while trying to answer your question."

# class AgentState(TypedDict):
#     messages: Annotated[Sequence[BaseMessage], add_messages]

# response = llm.invoke("Why do parrots talk?")

# print(response.content)  # Expected output: An explanation of why parrots mimic human speech.