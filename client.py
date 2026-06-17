import asyncio

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient

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
    weather_response = await agent.ainvoke({"messages": "what is the weather in Bidar?"})

    print(weather_response['messages'][-1].content)
    print(math_response['messages'][-1].content)


if __name__ == "__main__":
    asyncio.run(main())
