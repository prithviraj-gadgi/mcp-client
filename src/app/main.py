import asyncio

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_ollama import ChatOllama

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
    model = ChatOllama(model="gemma4:e4b")
    tools = await client.get_tools()
    agent = create_agent(model=model, tools=tools)
    math_response = await agent.ainvoke({"messages": "what's (3 + 5) x 12?"})
    weather_response = await agent.ainvoke({"messages": "what is the weather in Bengaluru?"})

    print(weather_response['messages'][-1].content)
    print(math_response['messages'][-1].content)


if __name__ == "__main__":
    asyncio.run(main())
