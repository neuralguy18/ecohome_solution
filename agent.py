import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.prebuilt import create_react_agent
from langchain_core.exceptions import OutputParserException
from tools import TOOL_KIT

load_dotenv()


ENERGY_SYSTEM_INSTRUCTIONS = """
You are EcoHome Energy Advisor — an expert AI designed to analyze household
energy usage and provide smart, actionable recommendations.
"""


class Agent:
    def __init__(self, instructions: str = ENERGY_SYSTEM_INSTRUCTIONS,
                 model: str = "gpt-4o-mini"):

        self.llm = ChatOpenAI(
            model=model,
            temperature=0.0,
            api_key=os.getenv("OPENAI_API_KEY")
        )

        self.graph = create_react_agent(
            name="eco_energy_advisor",
            prompt=instructions,  # <-- FIXED
            model=self.llm,
            tools=TOOL_KIT,
        )   
  
        

    def invoke(self, question: str, location: str = None, context: str = None, return_raw: bool = False) -> str:
        """
        Ask the Energy Advisor a question about energy optimization.

        Args:
            question (str): The user's question
            location (str, optional): Location for weather and pricing context
            context (str, optional): Additional system context
        
        Returns:
            str: The advisor's response with recommendations
        """

        sys_context_parts = []

        if location:
            sys_context_parts.append(f"Location: {location}")

        if context:
            sys_context_parts.append(context)

        messages = []

        # Add system context if any exists
        if sys_context_parts:
            messages.append(
                SystemMessage(content="\n".join(sys_context_parts))
            )

        # Add the main user request
        messages.append(
            HumanMessage(content=question)
        )

        # Get response from the agent graph
        raw_response = self.graph.invoke(
            {"messages": messages}
        )
        # If debugging tools → return full graph output
        if return_raw:
            return raw_response

        # Extract only the final assistant message text
        return raw_response["messages"][-1].content


    def get_agent_tools(self):
        """Get list of available tools for the Energy Advisor"""
        return [t.name for t in TOOL_KIT]

