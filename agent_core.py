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
from utils import extract_interactive_elements, truncate_text , find_element_ref , analyze_goal , extract_form_fields , _is_interactive_line
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
    
    def _prepare_invoke_messages(self, messages, keep_last: int = 5):
        """Prepare a structured sequence of messages for LLM.
        
        Structure:
        1. System message (context + tools)
        2. User's original goal
        3. Most recent actions/results
        """
        prepared = []

        # 1. Build system message with current context
        page_summary = ""
        if self.state.page_state:
            page_summary = f"\nCurrent page: {self.state.current_url or 'Unknown'}\n"
            if "title" in self.state.page_state:
                page_summary += f"Title: {self.state.page_state.get('title')}\n"
            if "buttons" in self.state.page_state:
                buttons = self.state.page_state["buttons"]
                page_summary += f"Buttons: {', '.join(buttons[:5])}\n"
            if "element_refs" in self.state.page_state:
                refs = self.state.page_state["element_refs"]
                ref_items = []
                for label, rlist in list(refs.items())[:5]:
                    ref_items.append(f"'{label}': {','.join(rlist)}")
                page_summary += f"Element refs: {', '.join(ref_items)}\n"

        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
            page_context=page_summary,
            step_count=f"{self.state.step_count}/{self.state.max_steps}"
        )
        prepared.append(SystemMessage(content=system_prompt))

        # 2. Include original goal (first human message)
        first_human = next((m for m in messages if isinstance(m, HumanMessage)), None)
        if first_human:
            prepared.append(first_human)

        # 3. Add last few non-system messages (actions/results)
        recent = []
        for msg in reversed(messages):
            if not isinstance(msg, SystemMessage):
                if len(recent) < keep_last:
                    recent.append(msg)
        recent.reverse()
        prepared.extend(recent)

        return prepared
    
    def planner_node(self, state: MessagesState):
        """Plan next action based on current state."""
        messages = state["messages"]
        
        # Check if we've reached step limit
        if self.state.step_count >= self.state.max_steps:
            return {"messages": messages + [AIMessage(content="Reached maximum step limit. Stopping execution.")]}
        
        try:
            # Prepare a structured sequence of messages
            invoke_messages = self._prepare_invoke_messages(messages, keep_last=5)

            # Call LLM with structured messages
            response = self.llm.invoke(invoke_messages)

            # If LLM returned tool calls but no content, add descriptive content
            if hasattr(response, "tool_calls") and response.tool_calls and not getattr(response, "content", None):
                tools_desc = "; ".join([f"{tc.get('name')}({tc.get('args')})" for tc in response.tool_calls])
                response.content = f"Planning next steps: {tools_desc}"

            self.state.step_count += 1
            # Add both the rebuilt system prompt and response to history
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
        
        # Track consecutive failures of the same type
        recent_errors = [msg for msg in messages[-3:] if isinstance(msg, ToolMessage) and "Error executing" in msg.content]
        similar_errors = len([msg for msg in recent_errors if "browser_click" in msg.content])
        
        for tool_call in getattr(last_message, "tool_calls", []) or []:
            tool_name = tool_call.get("name")
            tool_args = tool_call.get("args", {})
            tool_id = tool_call.get("id") or tool_name
            
            # Print tool call information
            print(f"\n Executing tool: {tool_name}")
            print(f"   - Arguments: {tool_args}")
            print(f"   - Call ID: {tool_id}")
            
            # Handle click operations
            if tool_name == "browser_click":
                # Always try to find the element in the latest snapshot
                if self.state.page_state and "snapshot_text" in self.state.page_state:
                    # Get element text, trying different possible keys
                    element_text = tool_args.get("element") or tool_args.get("selector", "") or tool_args.get("ref", "")
                    
                    # If we only have a ref, try to find its text from page state
                    if element_text.startswith("e") and "refs" in self.state.page_state:
                        for label, refs in self.state.page_state.get("refs", {}).items():
                            if element_text in refs:
                                element_text = label
                                break
                    
                    # Try to find ref
                    new_ref = find_element_ref(
                        self.state.page_state["snapshot_text"], 
                        element_text,
                        tool_args.get("element_type", "button")
                    )
                    
                    if new_ref:
                        # Use both element and ref for better reliability
                        tool_args["ref"] = new_ref
                        tool_args["element"] = element_text
                
            # Handle navigation
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
            
            try:
                # Print before execution
                print(f"   - Executing {tool_name} with args: {tool_args}")
                
                try:
                    result = await self.tool_manager.execute_tool(tool_name, tool_args)
                    print(f"   - {tool_name} executed successfully")
                except Exception as e:
                    print(f"   - Error executing {tool_name}: {str(e)}")
                    raise
                
                # Special handling for snapshot
                if tool_name == "browser_snapshot":
                    # Store the raw snapshot text for element finding
                    self.state.page_state["snapshot_text"] = result
                    page_elements = extract_interactive_elements(result)
                    self.state.page_state.update(page_elements)
                    
                    # Enhanced summary
                    summary = f"Page Snapshot Summary:\n"
                    summary += f"Title: {page_elements.get('title', 'Unknown')}\n"
                    summary += f"URL: {self.state.current_url or 'Unknown'}\n"
                    
                    # Show elements with their references
                    for element_type in ["buttons", "tabs", "inputs", "links"]:
                        if element_type in page_elements and page_elements[element_type]:
                            elements_with_refs = []
                            for element in page_elements[element_type][:8]:
                                refs = page_elements.get("refs", {}).get(element, [])
                                ref_str = f" (ref: {refs[0]})" if refs else ""
                                elements_with_refs.append(f'"{element}"{ref_str}')
                            summary += f"{element_type.capitalize()}: {', '.join(elements_with_refs)}\n"
                    
                    print(f"   - Snapshot summary generated")
                    print(f"   - Title: {page_elements.get('title', 'Unknown')}")
                    print(f"   - Buttons: {len(page_elements.get('buttons', []))}")
                    print(f"   - Inputs: {len(page_elements.get('inputs', []))}")
                    results.append(ToolMessage(content=summary, tool_call_id=tool_id))
                else:
                    result_str = truncate_text(str(result), 2000)
                    print(f"   - Tool result: {result_str[:200]}..." if len(result_str) > 200 else f"   - Tool result: {result_str}")
                    results.append(ToolMessage(
                        content=result_str,
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
        print(state['messages'])

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