import asyncio

from agents import set_trace_processors
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langsmith.integrations.openai_agents_sdk import OpenAIAgentsTracingProcessor

load_dotenv()

client = MultiServerMCPClient(
    {
        "math": {
            "url": "http://localhost:8081/mcp",
            "transport": "streamable-http",
        },
        "weather": {
            "url": "http://localhost:8080/mcp",
            "transport": "streamable-http",
        }
    }
)


async def main():
    tools = await client.get_tools()
    agent = create_agent(model="openai:gpt-4.1", tools=tools)
    math_response = await agent.ainvoke({"messages": "what's (3 + 5) x 12?"})
    weather_response = await agent.ainvoke({"messages": "what is the weather in Bidar Karnataka?"})

    print(weather_response['messages'][-1].content)
    print(math_response['messages'][-1].content)


if __name__ == "__main__":
    set_trace_processors([OpenAIAgentsTracingProcessor()])
    asyncio.run(main())
