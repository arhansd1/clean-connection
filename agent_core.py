"""Core agent logic and orchestration."""
import asyncio
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.graph import StateGraph, END
from langgraph.graph import START
from langgraph.graph import MessagesState

from tool_manager import ToolManager
from utils import extract_interactive_elements, truncate_text
from prompts import SYSTEM_PROMPT_TEMPLATE, REFLECTION_PROMPT

import base64
import os
from pathlib import Path

@dataclass
class AgentState:
    """Track agent execution state beyond just messages."""
    messages: List[Any] = field(default_factory=list)
    navigation_history: List[str] = field(default_factory=list)
    current_url: Optional[str] = None
    visited_elements: Set[str] = field(default_factory=set)
    page_state: Dict[str, Any] = field(default_factory=dict)
    step_count: int = 0
    max_steps: int = 100
    errors: List[str] = field(default_factory=list)
    task_complete: bool = False
 
    @property
    def last_message(self):
        return self.messages[-1] if self.messages else None


class WebAgent:
    """Intelligent web automation agent."""
    
    def __init__(self, llm, tool_manager: ToolManager):
        self.llm = llm
        self.tool_manager = tool_manager
        self.state = AgentState()

    def build_workflow(self):
        """Construct the agent workflow graph."""
        
        workflow = StateGraph(MessagesState)
        
        # Define nodes
        workflow.add_node("planner", self.planner_node)
        workflow.add_node("executor", self.executor_node)
        workflow.add_node("reflector", self.reflector_node)
        
        # Define edges
        workflow.add_edge(START, "planner")
        workflow.add_conditional_edges(
            "planner",
            self.route_from_planner,
            {
                "executor": "executor", 
                "reflector": "reflector",
                "exit": END
            }
        )
        workflow.add_edge("executor", "reflector")
        workflow.add_edge("reflector", "planner")
        
        return workflow.compile()
    
    def planner_node(self, state: MessagesState):
        """Plan next action based on current state."""
        
        messages = state["messages"]
        
        # Check if we've reached step limit
        if self.state.step_count >= self.state.max_steps:
            return {"messages": messages + [AIMessage(content="Reached maximum step count. Stopping execution.")]}
        
        # Generate task-specific context additions
        page_summary = ""
        if self.state.page_state:
            page_summary = f"\nCurrent page: {self.state.current_url or 'Unknown'}\n"
            if "title" in self.state.page_state:
                page_summary += f"Title: {self.state.page_state.get('title')}\n"
            if "interactives" in self.state.page_state:
                interactives = self.state.page_state["interactives"]
                page_summary += f"Available interactions: {', '.join(interactives[:10])}\n"
        
        # Create system message with context
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
            tools=self.tool_manager.get_tool_descriptions(),
            page_context=page_summary,
            step_count=f"{self.state.step_count}/{self.state.max_steps}"
        )
        
        system_msg = SystemMessage(content=system_prompt)
        
        
        
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [system_msg] + messages
            
        else:
            messages[0] = system_msg
            
        try:
            response = self.llm.invoke(messages)
            self.state.step_count += 1
            return {"messages": messages + [response]}
        except Exception as e:
            error_msg = f"Error in planning: {str(e)}"
            self.state.errors.append(error_msg)
            return {"messages": messages + [AIMessage(content=error_msg)]}
        
    
    async def executor_node(self, state: MessagesState):
        """Execute tools called by the planner."""
        messages = state["messages"]
        last_message = messages[-1]
        results = []
        print(last_message)
        for tool_call in getattr(last_message, "tool_calls", []) or []:
            tool_name = tool_call.get("name")
            tool_args = tool_call.get("args", {})
            tool_id = tool_call.get("id") or tool_name
            
            # Handle navigation history for loop detection
            print(tool_name)
            if tool_name == "browser_navigate" and "url" in tool_args:
                url = tool_args["url"]
                if url in self.state.navigation_history[-3:]:
                    results.append(ToolMessage(
                        content=f"Avoided navigation loop to {url}",
                        tool_call_id=tool_id
                    ))
                    continue
                self.state.navigation_history.append(url)
                self.state.current_url = url
            
            # Track clicked elements to avoid repetition
            if tool_name == "browser_click" and "element" in tool_args:
                element = tool_args["element"]
                self.state.visited_elements.add(element)
            
            try:
                result = await self.tool_manager.execute_tool(tool_name, tool_args)
                
                # Special handling for snapshot to extract page state
                if tool_name == "browser_snapshot":
                    page_elements = extract_interactive_elements(result)
                    self.state.page_state = page_elements
                    
                    # Create a more concise result for the agent
                    summary = f"Page Snapshot Summary:\n"
                    summary += f"Title: {page_elements.get('title', 'Unknown')}\n"
                    summary += f"URL: {self.state.current_url or 'Unknown'}\n"
                    summary += f"Headings: {', '.join(page_elements.get('headings', [])[:5])}\n"
                    summary += f"Buttons: {', '.join(page_elements.get('buttons', [])[:8])}\n"
                    summary += f"Inputs: {', '.join(page_elements.get('inputs', [])[:8])}\n"
                    summary += f"Links: {', '.join(page_elements.get('links', [])[:5])}\n"
                    
                    results.append(ToolMessage(content=summary, tool_call_id=tool_id))
                else:
                    results.append(ToolMessage(
                        content=truncate_text(result, 1500), 
                        tool_call_id=tool_id
                    ))
            except Exception as e:
                error_msg = f"Error executing {tool_name}: {str(e)}"
                self.state.errors.append(error_msg)
                results.append(ToolMessage(content=error_msg, tool_call_id=tool_id))
        
        return {"messages": messages + results}
    
    def reflector_node(self, state: MessagesState):
        """Reflect on execution results and determine if task is complete."""
        messages = state["messages"]
        
        # Look at recent tool results
        recent_tools = [m for m in messages[-5:] if isinstance(m, ToolMessage)]
        tool_contents = "\n".join([m.content for m in recent_tools])
        
        # Look for success signals in recent outputs
        success_signals = [
            ("screenshot", "saved", "Task involves taking screenshot"),
            ("form", "submitted", "Task involves form submission"),
            ("success", "confirmation", "Task received confirmation"),
            ("download", "completed", "Task involves downloading"),
            ("extracted", "data", "Task involves data extraction")
        ]
        
        # Check for completion signals
        completion_detected = False
        completion_reason = ""
        
        for signal, confirmation, reason in success_signals:
            if (signal.lower() in tool_contents.lower() and 
                confirmation.lower() in tool_contents.lower()):
                completion_detected = True
                completion_reason = reason
                break
        
        # Self-reflection message
        reflection = f"""
REFLECTION:
Step {self.state.step_count}/{self.state.max_steps}
Recent results: {truncate_text(tool_contents, 300)}
Visited elements: {len(self.state.visited_elements)}
Navigation history: {len(self.state.navigation_history)} pages
Errors: {len(self.state.errors)}
Task complete: {completion_detected}
"""
        
        if completion_detected:
            reflection += f"\nTask appears complete: {completion_reason}"
            self.state.task_complete = True
            return {"messages": messages + [AIMessage(content=reflection)]}
            
        return {"messages": messages}
    
    def route_from_planner(self, state: MessagesState):
        """Determine next state after planning."""
        messages = state["messages"]
        last_message = messages[-1]
        
        # Route to exit if task complete was flagged
        if self.state.task_complete:
            return "exit"
            
        # Route to executor if there are tool calls
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "executor"
            
        # Route to reflector for messages without tool calls
        # This allows for task completion detection
        return "reflector"

    async def run(self, goal: str):
        """Run the agent with the given goal."""
        print("[DEBUG] Building workflow...")
        workflow = self.build_workflow()
        
        print("[DEBUG] Initializing state...")
        initial_state = MessagesState(messages=[HumanMessage(content=goal)])
        
        print("[DEBUG] Invoking workflow...")
        final_state = await workflow.ainvoke(initial_state)
        
        print("[DEBUG] Workflow finished.")
        return final_state["messages"]