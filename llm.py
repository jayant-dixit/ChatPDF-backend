from langchain_core.messages import BaseMessage
from typing import TypedDict, Sequence, Annotated
from operator import add as add_messages
from langchain_groq import ChatGroq
from config import GROQ_API_KEY

# model = init_chat_model("google_genai:gemini-2.5-flash-lite", 
#                         api_key=GEMINI_API_KEY,
#                         temperature=0.2,
#                         max_output_tokens=512,
#                         top_p=0.8
#                         )

llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model = "groq/compound",
    temperature=0.2,
    max_tokens=512,
    timeout=None,
)

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

# response = llm.invoke("Why do parrots talk?")

# print(response.content)  # Expected output: An explanation of why parrots mimic human speech.