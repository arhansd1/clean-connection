import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def main():
    print("Before context")
    params = StdioServerParameters(
        command="/Users/arhan/.nvm/versions/node/v22.18.0/bin/npx",
        args=["-y", "@playwright/mcp@latest"]
    )
    try:
        async with stdio_client(params) as (read, write):
            print("Subprocess started")
            session = ClientSession(read, write)
            await session.initialize()
            print("Session initialized")
            tools = await session.list_tools()
            print("Available tools:", [t.name for t in tools.tools])
            await session.close()
    except Exception as e:
        print(f"Exception: {e}")

print("Before asyncio.run")
asyncio.run(main())
print("After asyncio.run")


 