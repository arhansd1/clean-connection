"""Main entry point for web automation agent."""
import asyncio
import os
from typing import Dict, Any, Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from agent_core import WebAgent
from tool_manager import ToolManager
from utils import analyze_goal

import os
os.environ["GOOGLE_API_KEY"] = "AIzaSyDJMNXFV6k6DbZlb-Z-mfEynssltlGnfFA"  # Replace with your actual API key
#AIzaSyA1S0_PviUC6e9gSzfXNNIu16LsODM_O0E
google_key = os.getenv("GOOGLE_API_KEY")


async def run_automation(goal: str, config: Optional[Dict[str, Any]] = None):
    """Run web automation with the given goal."""
    config = config or {}
    
    # Set up MCP client with headless mode and better browser management
    server_params = StdioServerParameters(
        command="/Users/arhan/.nvm/versions/node/v22.18.0/bin/npx",
        args=["-y", "@playwright/mcp@latest", "--isolated"],
    )
    
    print("🔌 Connecting to Playwright MCP server...")
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            print("✅ MCP session initialized")
            
            # Initialize tool manager
            tool_manager = ToolManager(session)
            await tool_manager.initialize_tools()
            print(f"🧰 Loaded {len(tool_manager.tools)} tools")
            
            # Get API key from environment or config
            api_key = config.get("api_key") or os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("Missing Google API key. Set GOOGLE_API_KEY environment variable.")
                
            # Initialize LLM
            model_name = config.get("model_name") or os.getenv("MODEL_NAME", "gemini-2.5-flash")
            llm = ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=api_key,
                temperature=0.2,
            ).bind_tools(tool_manager.tools)
            
            # Create and run agent
            agent = WebAgent(llm, tool_manager)
            
            # Analyze goal for better context
            goal_analysis = analyze_goal(goal)
            print(f"🎯 Goal analysis: {goal_analysis}")
            
            # Run the agent
            print(f"🤖 Starting automation for goal: {goal}")
    #        messages = await agent.run(goal)
           
            try:
                # Run the agent with just the goal parameter
                messages = await agent.run(goal)
            except asyncio.TimeoutError:
                print("❌ Agent run timed out!")
                return []
            except Exception as e:
                print(f"❌ Agent run failed: {str(e)}")
                return []
                
            # Print summary
            print("\n==== Automation Complete ====")
            last_msg = messages[-1] if messages else None
            if last_msg:
                print(f"Final status: {getattr(last_msg, 'content', 'No content')}")
            
            # Return final messages for analysis
            return messages


async def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Web Automation Agent")
    parser.add_argument("goal", nargs="*", help="User goal for web automation")
    parser.add_argument("--api-key", help="Google API key (overrides environment variable)")
    parser.add_argument("--model", help="Model name (default: gemini-2.5-flash)")
    parser.add_argument("--max-steps", type=int, default=50, help="Maximum execution steps")
    
    args = parser.parse_args()
   
    # Default goal if none provided
    goal = "".join(args.goal) if args.goal else (
        "Navigate to https://form.jotform.com/252131352654450 , get all the job query form fields and fill them with dummy data and submit the form. "
    )

    config = {
        "api_key": args.api_key,
        "model_name": args.model,
        "max_steps": args.max_steps
    }
    
    await run_automation(goal, config)


if __name__ == "__main__":
    asyncio.run(main())

#https://form.jotform.com/252131352654450
#https://job.10xscale.ai/4846461985313787904
