"""Simplified agent core with clean 3-node architecture."""
import json
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, START, MessagesState

from tool_manager import ToolManager
from utils import extract_form_fields, match_user_data_to_fields
from prompts import PLANNER_PROMPT, FILLER_PROMPT, EXECUTOR_PROMPT

@dataclass
class AgentState:
    """Simple agent state."""
    messages: List[Any] = field(default_factory=list)
    step_count: int = 0
    max_steps: int = 50
    current_snapshot: str = ""
    unfilled_attempts: int = 0
    task_complete: bool = False

class WebAgent:
    """Simplified web automation agent with 3-node architecture."""
    
    def __init__(self, llm, tool_manager: ToolManager, user_data: Dict[str, str]):
        self.llm = llm
        self.tool_manager = tool_manager
        self.user_data = user_data
        self.state = AgentState()

    def build_workflow(self):
        """Build the simplified 3-node workflow."""
        workflow = StateGraph(MessagesState)
        
        workflow.add_node("planner", self.planner_node)
        workflow.add_node("filler", self.filler_node)
        workflow.add_node("executor", self.executor_node)

        workflow.add_edge(START, "planner")
        workflow.add_conditional_edges(
            "planner",
            self.route_from_planner,
            {"filler": "filler", "executor": "executor", "end": "__end__"}
        )
        workflow.add_edge("filler", "executor")
        workflow.add_edge("executor", "planner")
        
        return workflow.compile()
    
    def planner_node(self, state: MessagesState):
        """Planner: Takes snapshots and decides next action."""
        messages = state["messages"]
        
        if self.state.step_count >= self.state.max_steps:
            return {"messages": messages + [AIMessage(content="Task completed: Reached maximum steps")]}
        
        # Check if this is initial navigation
        if self.state.step_count == 0:
            goal = messages[0].content if messages else ""
            if "http" in goal:
                import re
                url_match = re.search(r'https?://[^\s]+', goal)
                if url_match:
                    url = url_match.group(0)
                    response = AIMessage(
                        content=f"Navigating to {url}",
                        tool_calls=[{
                            "name": "browser_navigate",
                            "args": {"url": url},
                            "id": "nav_1"
                        }]
                    )
                    self.state.step_count += 1
                    return {"messages": messages + [response]}
        
        # Get tool descriptions for prompt
        tool_descriptions = []
        for tool_name, schema in self.tool_manager.tool_schemas.items():
            tool_descriptions.append(f"- {tool_name}: {schema['description']}")
        tools_text = "\n".join(tool_descriptions)
        
        # Create system message with planner prompt
        system_prompt = PLANNER_PROMPT.format(
            tools=tools_text,
            step_count=f"{self.state.step_count}/{self.state.max_steps}"
        )
        
        # Prepare messages for LLM
        planner_messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content="Analyze current page state and decide next action")
        ]
        
        # Add recent context
        recent_messages = [msg for msg in messages[-5:] if isinstance(msg, (AIMessage, ToolMessage))]
        planner_messages.extend(recent_messages)
        
        try:
            response = self.llm.invoke(planner_messages)
            if not response.content:
                response.content = "Taking snapshot to analyze page"
            
            # Default to snapshot if no tool calls
            if not getattr(response, 'tool_calls', []):
                response.tool_calls = [{
                    "name": "browser_snapshot",
                    "args": {},
                    "id": f"snap_{self.state.step_count}"
                }]
            
            self.state.step_count += 1
            return {"messages": messages + [response]}
            
        except Exception as e:
            error_response = AIMessage(content=f"Planner error: {str(e)}")
            return {"messages": messages + [error_response]}
    
    def route_from_planner(self, state: MessagesState):
        """Route based on current page analysis."""
        messages = state["messages"]
        
        # Check if we have a recent snapshot
        recent_snapshots = [msg for msg in messages[-5:] if isinstance(msg, ToolMessage) and "Page Snapshot" in str(msg.content)]
        if not recent_snapshots:
            return "executor"  # Need to execute snapshot first
        
        latest_snapshot = recent_snapshots[-1].content
        self.state.current_snapshot = latest_snapshot
        
        # Check for form fields
        form_fields = extract_form_fields(latest_snapshot)
        
        # Check for submit success
        if any(phrase in latest_snapshot.lower() for phrase in ["submitted", "thank you", "success", "received"]):
            print("✅ Form submitted successfully!")
            self.state.task_complete = True
            return "end"
        
        # If we have form fields, route to filler
        if form_fields:
            print(f"📝 Found {len(form_fields)} form fields")
            return "filler"
        
        # Look for next/submit buttons
        if any(btn in latest_snapshot.lower() for btn in ["submit", "next", "continue", "apply"]):
            print("🔘 Found action buttons")
            return "executor"
        
        return "executor"
    
    def filler_node(self, state: MessagesState):
        """Filler: Matches user data to form fields."""
        messages = state["messages"]
        
        if not self.state.current_snapshot:
            return {"messages": messages + [AIMessage(content="No snapshot available for form analysis")]}
        
        # Extract form fields from snapshot
        form_fields = extract_form_fields(self.state.current_snapshot)
        if not form_fields:
            return {"messages": messages + [AIMessage(content="No form fields found to fill")]}
        
        # Match user data to fields
        matched_fields = match_user_data_to_fields(form_fields, self.user_data)
        
        if not matched_fields:
            # If no matches, check if we've tried before and increment attempts
            self.state.unfilled_attempts += 1
            if self.state.unfilled_attempts >= 2:
                return {"messages": messages + [AIMessage(content="Could not match user data to fields after 2 attempts, continuing...")]}
            return {"messages": messages + [AIMessage(content="No matching data found for form fields, will retry")]}
        
        # Reset unfilled attempts on successful match
        self.state.unfilled_attempts = 0
        
        # Format user data and form fields for prompt
        user_data_text = "\n".join([f"- {k}: {v}" for k, v in self.user_data.items()])
        form_fields_text = "\n".join([f"- {f['label']} ({f['type']}) ref:{f['ref']}" for f in form_fields])
        
        # Create system message with filler prompt
        system_prompt = FILLER_PROMPT.format(
            user_data=user_data_text,
            form_fields=form_fields_text
        )
        
        # Prepare messages for LLM
        filler_messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Fill {len(matched_fields)} form fields with matching user data")
        ]
        
        # Create tool calls for filling fields
        tool_calls = []
        for field in matched_fields:
            if field["action"] == "type":
                tool_calls.append({
                    "name": "browser_type",
                    "args": {
                        "element": field["label"],
                        "ref": field["ref"],
                        "text": field["value"]
                    },
                    "id": f"type_{field['ref']}"
                })
            elif field["action"] == "select":
                tool_calls.append({
                    "name": "browser_select_option",
                    "args": {
                        "element": field["label"],
                        "ref": field["ref"],
                        "values": [field["value"]]
                    },
                    "id": f"select_{field['ref']}"
                })
            elif field["action"] == "upload":
                tool_calls.append({
                    "name": "browser_file_upload",
                    "args": {
                        "paths": ["/Users/arhan/Desktop/clean-connection 2/sample.pdf"],
                        "ref": field["ref"]
                    },
                    "id": f"upload_{field['ref']}"
                })
            elif field["action"] == "click":
                tool_calls.append({
                    "name": "browser_click",
                    "args": {
                        "element": field["label"],
                        "ref": field["ref"]
                    },
                    "id": f"click_{field['ref']}"
                })
        
        response = AIMessage(
            content=f"Filling {len(matched_fields)} form fields with user data",
            tool_calls=tool_calls
        )
        
        return {"messages": messages + [response]}
    
    async def executor_node(self, state: MessagesState):
        """Executor: Executes tool calls and handles actions."""
        messages = state["messages"]
        last_message = messages[-1]
        results = []
        
        # If no tool calls, look for action buttons
        if not getattr(last_message, "tool_calls", []):
            if self.state.current_snapshot:
                # Look for submit or next buttons - prioritize "next" over "submit"
                button_found = False
                lines = self.state.current_snapshot.split('\n')
                
                # First pass: look for "next" buttons
                for line in lines:
                    line_lower = line.lower()
                    if ("next" in line_lower or "continue" in line_lower) and "button" in line_lower:
                        ref_match = re.search(r'\[ref=(e\d+)\]', line)
                        if ref_match:
                            ref = ref_match.group(1)
                            try:
                                result = await self.tool_manager.execute_tool("browser_click", {
                                    "element": "Next",
                                    "ref": ref
                                })
                                results.append(ToolMessage(
                                    content=f"Clicked next button: {result}",
                                    tool_call_id="next_click"
                                ))
                                print("Clicked Next button")
                                button_found = True
                                break
                            except Exception as e:
                                results.append(ToolMessage(
                                    content=f"Failed to click next: {str(e)}",
                                    tool_call_id="next_click_error"
                                ))
                
                # Second pass: look for submit buttons if no next found
                if not button_found:
                    for line in lines:
                        line_lower = line.lower()
                        if ("submit" in line_lower or "apply" in line_lower) and "button" in line_lower:
                            ref_match = re.search(r'\[ref=(e\d+)\]', line)
                            if ref_match:
                                ref = ref_match.group(1)
                                try:
                                    result = await self.tool_manager.execute_tool("browser_click", {
                                        "element": "Submit",
                                        "ref": ref
                                    })
                                    results.append(ToolMessage(
                                        content=f"Clicked submit button: {result}",
                                        tool_call_id="submit_click"
                                    ))
                                    print("Clicked Submit button")
                                    button_found = True
                                    break
                                except Exception as e:
                                    results.append(ToolMessage(
                                        content=f"Failed to click submit: {str(e)}",
                                        tool_call_id="submit_click_error"
                                    ))
                
                if not button_found:
                    results.append(ToolMessage(
                        content="No action buttons found to click",
                        tool_call_id="no_action"
                    ))
            else:
                results.append(ToolMessage(
                    content="No snapshot available for action",
                    tool_call_id="no_snapshot"
                ))
                
            return {"messages": messages + results}
        
        # Execute tool calls
        for tool_call in getattr(last_message, "tool_calls", []):
            tool_name = tool_call.get("name")
            tool_args = tool_call.get("args", {})
            tool_id = tool_call.get("id") or tool_name
            
            print(f"Executing: {tool_name} with {tool_args}")
            
            try:
                result = await self.tool_manager.execute_tool(tool_name, tool_args)
                
                # Handle snapshot results specially
                if tool_name == "browser_snapshot":
                    self.state.current_snapshot = result
                    # Extract basic counts for summary
                    button_count = len([line for line in result.split('\n') if 'button' in line.lower()])
                    input_count = len([line for line in result.split('\n') if any(t in line.lower() for t in ['textbox', 'input', 'combobox'])])
                    summary = f"Page snapshot taken - {button_count} buttons, {input_count} input fields found"
                    results.append(ToolMessage(content=summary, tool_call_id=tool_id))
                else:
                    # Truncate long results
                    result_text = str(result)
                    if len(result_text) > 500:
                        result_text = result_text[:500] + "... [truncated]"
                    results.append(ToolMessage(content=result_text, tool_call_id=tool_id))
                    
            except Exception as e:
                error_msg = f"Error executing {tool_name}: {str(e)}"
                results.append(ToolMessage(content=error_msg, tool_call_id=tool_id))
                print(f"Error: {error_msg}")
        
        return {"messages": messages + results}
    
    async def run(self, goal: str):
        """Run the simplified agent."""
        print("🔄 Building workflow...")
        workflow = self.build_workflow()
        
        print("🚀 Starting workflow...")
        initial_state = MessagesState(messages=[HumanMessage(content=goal)])
        
        final_state = await workflow.ainvoke(initial_state)
        return final_state["messages"]