import asyncio
import json
import os
from typing import Any, Dict, List, Optional, TypedDict
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
# from langgraph.prebuilt import ToolExecutor, ToolInvocation
import nest_asyncio
import os
os.environ["GOOGLE_APT_KEY"] = "AIzaSyDJMNXFV6k6DbZlb-Z-mfEynssltlGnfFA"  # Replace with your actual API key

# Apply nest_asyncio for Jupyter compatibility
nest_asyncio.apply()

# Agent State Definition
class WebAgentState(TypedDict):
    query: str
    messages: List[Any]
    next_action: Dict[str, Any]
    browser_context: str
    task_complete: bool
    step_count: int

class SimpleWebAgent:
    def __init__(self, google_api_key: str = None):
        self.mcp_client = None
        self.session = None
        self.available_tools = []
        
        # Initialize Gemini LLM
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-pro",
            google_api_key=google_api_key or os.getenv("GOOGLE_API_KEY"),
            temperature=0.1,
            verbose=True
        )
        
        # LangGraph workflow
        self.workflow = None
        self.app = None

    async def connect_mcp(self):
        """Connect to Playwright MCP server"""
        print("🔌 Connecting to Playwright MCP server...")
        server_params = StdioServerParameters(
            command = "/Users/arhan/.nvm/versions/node/v22.18.0/bin/npx",
            args=["-y", "@playwright/mcp@latest"]
        )
        print(f"🛠️  MCP command: {server_params.command} {server_params.args}")
        try:
            async with stdio_client(server_params) as (read, write):
                print("✅ MCP subprocess started")
                self.read = read
                self.write = write
                self.session = ClientSession(self.read, self.write)
                await self.session.initialize()
            print("✅ MCP session initialized")
            # Get available tools
            tools_response = await self.session.list_tools()
            self.available_tools = [tool.name for tool in tools_response.tools]
            print(f"✅ Connected! Available tools: {len(self.available_tools)}")
            return True
        except Exception as e:
            print(f"❌ MCP connection failed: {e}")
            raise

    async def call_mcp_tool(self, tool_name: str, args: Dict[str, Any]) -> str:
        """Call Playwright MCP tool"""
        print(f"🔧 Calling {tool_name} with args: {args}")
        
        result = await self.session.call_tool(tool_name, args)
        
        if result.content and hasattr(result.content[0], "text"):
            return result.content[0].text
        return str(result)

    def build_workflow(self):
        """Build simple LangGraph workflow"""
        workflow = StateGraph(WebAgentState)
        
        # Add nodes
        workflow.add_node("planner", self.plan_action)
        workflow.add_node("executor", self.execute_action)
        workflow.add_node("evaluator", self.evaluate_progress)
        
        # Set entry point
        workflow.set_entry_point("planner")
        
        # Add edges
        workflow.add_conditional_edges(
            "planner",
            lambda state: "execute" if state["next_action"] else "end",
            {"execute": "executor", "end": END}
        )
        
        workflow.add_conditional_edges(
            "executor", 
            lambda state: "evaluate" if not state["task_complete"] else "end",
            {"evaluate": "evaluator", "end": END}
        )
        
        workflow.add_conditional_edges(
            "evaluator",
            lambda state: "end" if state["task_complete"] or state["step_count"] > 10 else "planner",
            {"end": END, "planner": "planner"}
        )
        
        self.app = workflow.compile()

    async def plan_action(self, state: WebAgentState) -> WebAgentState:
        """LLM plans next action based on query and browser context"""
        print(f"\n🧠 PLANNING (Step {state['step_count'] + 1})...")
        
        system_prompt = f"""You are a web automation agent with access to Playwright browser tools.

Available tools: {', '.join(self.available_tools)}

Current browser context: {state.get('browser_context', 'No browser open yet')}

User wants: {state['query']}

Plan the next single action. Respond ONLY with JSON:
{{
    "tool": "tool_name",
    "args": {{"param": "value"}},
    "reasoning": "why this action"
}}

Common tools:
- mcp_playwright_browser_navigate: Navigate to URL
- mcp_playwright_browser_click: Click elements  
- mcp_playwright_browser_type: Type text
- mcp_playwright_browser_snapshot: Get page state
- mcp_playwright_browser_take_screenshot: Take screenshot
"""

        try:
            response = await self.llm.ainvoke([SystemMessage(content=system_prompt)])
            plan = json.loads(response.content)
            
            state["next_action"] = plan
            state["step_count"] = state.get("step_count", 0) + 1
            
            print(f"📋 PLAN: {plan['reasoning']}")
            print(f"🎯 ACTION: {plan['tool']} {plan['args']}")
            
        except Exception as e:
            print(f"❌ Planning failed: {e}")
            state["next_action"] = {}
            
        return state

    async def execute_action(self, state: WebAgentState) -> WebAgentState:
        """Execute the planned action"""
        action = state["next_action"]
        
        if not action:
            state["task_complete"] = True
            return state
            
        print(f"\n⚡ EXECUTING: {action['tool']}")
        
        try:
            result = await self.call_mcp_tool(action["tool"], action["args"])
            
            # Update browser context if we got page info
            if "Page state" in result or "Page URL" in result:
                state["browser_context"] = result[:1000]  # Keep context manageable
                
            state["messages"].append(f"Executed {action['tool']}: Success")
            print("✅ Action completed")
            
        except Exception as e:
            print(f"❌ Execution failed: {e}")
            state["messages"].append(f"Failed {action['tool']}: {e}")
            
        return state

    async def evaluate_progress(self, state: WebAgentState) -> WebAgentState:
        """LLM evaluates if task is complete"""
        print(f"\n🔍 EVALUATING progress...")
        
        eval_prompt = f"""Based on the user query and current browser state, is the task complete?

User query: {state['query']}
Current browser context: {state.get('browser_context', 'No context')}
Steps taken: {state['step_count']}

Respond ONLY with JSON:
{{"complete": true/false, "reason": "explanation"}}
"""

        try:
            response = await self.llm.ainvoke([SystemMessage(content=eval_prompt)])
            evaluation = json.loads(response.content)
            
            state["task_complete"] = evaluation["complete"]
            print(f"📊 EVALUATION: {evaluation['reason']}")
            
        except Exception as e:
            print(f"❌ Evaluation failed: {e}")
            # Fallback: complete after many steps
            state["task_complete"] = state["step_count"] > 8
            
        return state

    async def run_task(self, query: str) -> Dict[str, Any]:
        """Run a web automation task"""
        print(f"\n🚀 STARTING TASK: {query}")
        print("=" * 60)
        
        # Initialize state
        initial_state = WebAgentState(
            query=query,
            messages=[],
            next_action={},
            browser_context="",
            task_complete=False,
            step_count=0
        )
        
        # Run workflow
        final_state = await self.app.ainvoke(initial_state)
        
        print(f"\n🎯 TASK COMPLETED in {final_state['step_count']} steps")
        
        return {
            "success": final_state["task_complete"],
            "query": query,
            "steps": final_state["step_count"],
            "messages": final_state["messages"],
            "browser_context": final_state.get("browser_context", "")
        }

    async def cleanup(self):
        """Cleanup MCP connection"""
        if self.session:
            await self.session.close()
            print("🧹 Cleaned up MCP connection")

# Initialize the simple agent
async def create_web_agent(google_api_key: str = None) -> SimpleWebAgent:
    """Create and initialize the web automation agent"""
    agent = SimpleWebAgent(google_api_key)
    await agent.connect_mcp()
    agent.build_workflow()
    return agent




async def demo_web_automation():
    """Demo the simple web automation agent"""
    
    # Get API key (set your Google API key here or in environment)
    google_api_key = os.getenv("GOOGLE_API_KEY")  # or "your-api-key-here"
    
    if not google_api_key:
        print("⚠️  No Google API key found. Set GOOGLE_API_KEY environment variable or pass directly.")
        print("🔧 Using fallback behavior without LLM planning...")
    
    # Create agent
    agent = await create_web_agent(google_api_key)
    
    try:
        # Test different web automation tasks
        test_queries = [
            "Go to Google and search for latest AI news",
            "Navigate to YouTube and search for Python tutorials", 
            "Visit GitHub and search for LangGraph repositories",
            "Go to Wikipedia and search for artificial intelligence",
            "Open Reddit and browse the programming subreddit"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n{'='*60}")
            print(f"🎯 TEST {i}: {query}")
            print('='*60)
            
            result = await agent.run_task(query)
            
            if result["success"]:
                print(f"✅ SUCCESS: Completed in {result['steps']} steps")
            else:
                print(f"⚠️  PARTIAL: Completed {result['steps']} steps")
                
            print(f"📝 Final context: {result['browser_context'][:200]}...")
            
            # Small delay between tasks
            await asyncio.sleep(2)
            
    finally:
        await agent.cleanup()

# Quick single task function
async def run_single_task(query: str, google_api_key: str = None):
    """Run a single web automation task"""
    agent = await create_web_agent(google_api_key)
    
    try:
        result = await agent.run_task(query)
        return result
    finally:
        await agent.cleanup()

# Interactive function for custom queries
async def interactive_web_automation():
    """Interactive web automation session"""
    google_api_key = os.getenv("AIzaSyDJMNXFV6k6DbZlb-Z-mfEynssltlGnfFA")
    agent = await create_web_agent(google_api_key)
    
    print("🤖 Interactive Web Automation Agent")
    print("Type your web automation requests. Type 'quit' to exit.")
    print("Examples:")
    print("  - 'Search Google for latest news'")  
    print("  - 'Go to YouTube and find cooking videos'")
    print("  - 'Navigate to Stack Overflow and search for Python help'")
    
    try:
        while True:
            print("\n" + "-"*50)
            query = input("🎯 Enter your task: ").strip()
            
            if query.lower() in ['quit', 'exit', 'stop']:
                break
                
            if query:
                result = await agent.run_task(query)
                print(f"\n📊 Result: {'Success' if result['success'] else 'Partial'}")
                
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    finally:
        await agent.cleanup()

print("🎯 Available functions:")
print("  - demo_web_automation(): Run predefined test cases")
print("  - run_single_task('your query'): Run one task")
print("  - interactive_web_automation(): Interactive session")

# Quick Demo - Simple Web Automation Agent
async def quick_demo():
    """Quick demonstration of the simple web automation agent"""
    
    print("🚀 SIMPLE WEB AUTOMATION AGENT DEMO")
    print("=" * 50)
    
    # Set your Google API key here or in environment variable
    google_api_key = "AIzaSyDJMNXFV6k6DbZlb-Z-mfEynssltlGnfFA"  # Replace with "your-google-api-key" or set GOOGLE_API_KEY env var

    # Create and test the agent
    agent = await create_web_agent(google_api_key)
    
    try:
        # Demo task 1: Google search (like we did manually before)
        print("\n🎯 DEMO 1: Google AI News Search")
        result1 = await agent.run_task("Go to Google and search for latest AI news")
        
        # Demo task 2: YouTube search  
        print("\n🎯 DEMO 2: YouTube Search")
        result2 = await agent.run_task("Navigate to YouTube and search for LangGraph tutorials")
        
        # Demo task 3: GitHub search
        print("\n🎯 DEMO 3: GitHub Search") 
        result3 = await agent.run_task("Go to GitHub and search for web automation projects")
        
        print("\n📊 DEMO SUMMARY:")
        print(f"✅ Task 1 Success: {result1['success']} ({result1['steps']} steps)")
        print(f"✅ Task 2 Success: {result2['success']} ({result2['steps']} steps)")  
        print(f"✅ Task 3 Success: {result3['success']} ({result3['steps']} steps)")
        
    finally:
        await agent.cleanup()
        print("\n🎉 Demo completed!")

# Run the quick demo
if __name__ == "__main__":
    import asyncio

    async def main():
        await quick_demo()
        # For custom tasks, uncomment and modify:
        await run_single_task("Your custom web automation task here")
        # For interactive session, uncomment:
        # await interactive_web_automation()
    asyncio.run(main())

    