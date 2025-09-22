"""Simplified main entry point for web automation agent."""
import asyncio
import os
from typing import Dict, Any, Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from agent_core import WebAgent
from tool_manager import ToolManager

# Set your API key
os.environ["GOOGLE_API_KEY"] = "AIzaSyDJMNXFV6k6DbZlb-Z-mfEynssltlGnfFA"

# User data for form filling
USER_DATA = {
    'name': 'John Doe',
    'first_name': 'John',
    'last_name': 'Doe',
    'email': 'john.doe@example.com',
    'phone': '+91 9876543210',
    'mobile': '+91 9876543210',
    'address': '123 Main Street, Bangalore, Karnataka 560001',
    'city': 'Bangalore',
    'state': 'Karnataka',
    'country': 'India',
    'zipcode': '560001',
    'postal_code': '560001',
    'experience': '5 years',
    'skills': 'Python, JavaScript, React, Node.js',
    'education': 'Bachelor of Technology in Computer Science',
    'company': 'Tech Solutions Pvt Ltd',
    'position': 'Senior Software Developer',
    'salary': '800000',
    'linkedin': 'https://linkedin.com/in/johndoe',
    'portfolio': 'https://johndoe.dev',
    'website': 'https://johndoe.dev',
    'github': 'https://github.com/johndoe',
    'cover_letter': 'I am excited to apply for this position and contribute my skills to your team.',
    'message': 'Looking forward to hearing from you.',
    'comments': 'Available to start immediately.',
    'availability': 'Immediate',
    'notice_period': '30 days',
    'expected_salary': '1000000',
    'current_salary': '800000'
}

async def run_automation(goal: str, user_data: Dict[str, str] = None):
    """Run web automation with the given goal."""
    user_data = user_data or USER_DATA
    
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
            
            # Initialize LLM
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("Missing Google API key")
                
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                google_api_key=api_key,
                temperature=0.2,
            ).bind_tools(tool_manager.tools)
            
            # Create and run agent
            agent = WebAgent(llm, tool_manager, user_data)
            
            print(f"🤖 Starting automation for goal: {goal}")
            try:
                messages = await agent.run(goal)
                print("\n==== Automation Complete ====")
                return messages
            except Exception as e:
                print(f"❌ Agent run failed: {str(e)}")
                return []

async def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Simplified Web Automation Agent")
    parser.add_argument("goal", nargs="*", help="User goal for web automation")
    
    args = parser.parse_args()
   
    goal = " ".join(args.goal) if args.goal else (
        "Navigate to https://form.jotform.com/252131352654450 and fill the form with my data and submit it."
    )
    
    await run_automation(goal, USER_DATA)

if __name__ == "__main__":
    asyncio.run(main())

#https://form.jotform.com/252131352654450
#https://job.10xscale.ai/4846461985313787904
