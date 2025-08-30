"""Core agent logic and orchestration."""
import asyncio
import json
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.graph import StateGraph
from langgraph.graph import START
from langgraph.graph import MessagesState

from tool_manager import ToolManager
from utils import extract_interactive_elements, truncate_text, find_element_ref, analyze_goal, extract_form_fields, _is_interactive_line, parse_form_fields_enhanced
from prompts import SYSTEM_PROMPT_TEMPLATE, FILLER_PROMPT_TEMPLATE

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
        workflow.add_node("filler", self.filler_node)

        # Define edges
        workflow.add_edge(START, "planner")
        workflow.add_conditional_edges(
            "planner",
            self.route_from_planner,
            {"executor": "executor", "filler": "filler"}
        )
        workflow.add_edge("executor", "planner")
        workflow.add_edge("filler", "executor")
        
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
                page_summary += f"Buttons ({len(buttons)}): {', '.join(buttons[:10])}"
                if len(buttons) > 10:
                    page_summary += f"... and {len(buttons) - 10} more"
                page_summary += "\n"
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
            print(response)
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
                # Unwrap kwargs if present
                if isinstance(tool_args, dict):
                    if "kwargs" in tool_args:
                        if isinstance(tool_args["kwargs"], str):
                            try:
                                tool_args = json.loads(tool_args["kwargs"])
                            except:
                                pass
                        else:
                            # Flatten the kwargs structure
                            tool_args = tool_args["kwargs"]
                
                # Ensure both element and ref are present for relevant tools
                if tool_name in ["browser_type", "browser_click", "browser_select_option"]:
                    # If we have ref but no element, try to find the element from page state
                    if "ref" in tool_args and "element" not in tool_args:
                        if self.state.page_state and "refs" in self.state.page_state:
                            for element, refs in self.state.page_state["refs"].items():
                                if tool_args["ref"] in refs:
                                    tool_args["element"] = element
                                    break
                
                # Print raw metadata before execution
                print(f"\n[TOOL_METADATA] name={tool_name} args={tool_args} call_id={tool_id}")
                
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
                    buttons = page_elements.get('buttons', [])
                    print(f"   - Buttons ({len(buttons)}): {', '.join(buttons[:5])}" + ("..." if len(buttons) > 5 else ""))
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
    
    

    def route_from_planner(self, state: MessagesState):
        """Determine next state after planning."""
        messages = state["messages"]
        last_message = messages[-1]

        # If we have page state, check for form indicators
        if self.state.page_state:
            form_indicators = 0
            
            # Check for multiple input fields
            inputs = self.state.page_state.get("inputs", [])
            if len(inputs) >= 2:  # At least two input fields
                form_indicators += 1
            
            # Check for common form field names
            common_fields = {"name", "email", "phone", "address", "password"}
            field_matches = sum(1 for field in inputs if any(common in field.lower() for common in common_fields))
            if field_matches >= 1:
                form_indicators += 1
            
            # Check for required fields (marked with *)
            required_fields = sum(1 for field in inputs if "*" in field)
            if required_fields > 0:
                form_indicators += 1
                
            # Check for submit/continue buttons along with inputs
            buttons = self.state.page_state.get("buttons", [])
            submit_buttons = any(b.lower() in ["submit", "continue", "next", "apply"] for b in buttons)
            if submit_buttons and inputs:
                form_indicators += 1
            
            # If we have enough form indicators and not in a filling loop
            if form_indicators >= 2:  # At least 2 indicators needed
                # Check if we're not already in a form-filling loop
                recent_fill_attempts = len([m for m in messages[-3:] if isinstance(m, ToolMessage) and "browser_type" in str(m.content)])
                if recent_fill_attempts < 3:  # Allow up to 3 consecutive fill attempts
                    return "filler"
        
        # Otherwise proceed to executor
        return "executor"

    def filler_node(self, state: MessagesState):
        """Specialized node for filling forms with enhanced field handling."""
        messages = state["messages"]
        
        # Create a form context with all the form fields, buttons, and their references
        form_fields = []
        interactive_buttons = []
        submission_buttons = []
        dropdowns = []
        radio_groups = []
        checkboxes = []
        
        if self.state.page_state:
            # Get all element references
            refs = self.state.page_state.get("refs", {})
            
            # Track which refs we've already added to avoid duplicates
            added_refs = set()
            
            # 1. Process regular input fields
            inputs = self.state.page_state.get("inputs", [])
            for input_field in inputs:
                ref = refs.get(input_field, [''])[0]
                if ref and ref not in added_refs:
                    # Format field with clean structure
                    field_lower = input_field.lower()
                    field_type = "text"
                    
                    if any(kw in field_lower for kw in ["email", "e-mail"]):
                        field_type = "email"
                    elif any(kw in field_lower for kw in ["phone", "mobile", "telephone"]):
                        field_type = "phone"
                    elif any(kw in field_lower for kw in ["date", "calendar"]):
                        field_type = "date"
                    elif any(kw in field_lower for kw in ["pay", "salary", "money"]):
                        field_type = "number"
                        
                    form_fields.append(f"- Text field, {input_field}, ref:{ref}, type:{field_type}")
                    added_refs.add(ref)
            
            # 2. Process dropdowns/selects with their options
            dropdowns_list = self.state.page_state.get("comboboxes", [])
            for dropdown in dropdowns_list:
                ref = refs.get(dropdown, [''])[0]
                if ref and ref not in added_refs:
                    # Look for options in refs under <dropdown>_options key
                    options_key = f"{dropdown}_options"
                    options = refs.get(options_key, [])
                    
                    # If no options found with that key, try to find options from snapshot
                    if not options and self.state.page_state and "snapshot_text" in self.state.page_state:
                        snapshot_lines = self.state.page_state["snapshot_text"].splitlines()
                        # Find the line with this dropdown and scan for options after it
                        for i, line in enumerate(snapshot_lines):
                            if dropdown in line and "combobox" in line.lower():
                                # Scan next lines for options
                                for j in range(i+1, min(len(snapshot_lines), i+15)):
                                    next_line = snapshot_lines[j].strip()
                                    if 'option' in next_line.lower() and '"' in next_line:
                                        # Extract option text
                                        option_match = re.search(r'"([^"]+)"', next_line)
                                        if option_match:
                                            option_text = option_match.group(1).strip()
                                            # Remove [selected] or other annotations
                                            option_text = re.sub(r'\s*\[.*\]$', '', option_text).strip()
                                            if option_text and option_text not in options:
                                                options.append(option_text)
                                    elif next_line and not next_line.startswith('-'):
                                        break
                                break
                    
                    options_str = f", options:[{','.join(options)}]" if options else ""
                    dropdowns.append(f"- Dropdown, {dropdown}, ref:{ref}{options_str}")
                    added_refs.add(ref)
            
            # 3. Process radio groups and radio buttons
            radio_groups_list = self.state.page_state.get("radio_groups", [])
            for radio in radio_groups_list:
                ref = refs.get(radio, [''])[0]
                if ref and ref not in added_refs:
                    # Check if this is a skill rating or yes/no question
                    radio_lower = radio.lower()
                    if any(skill in radio_lower for skill in ["microsoft", "communication", "seo", "skill"]):
                        # This is a skill rating, find the rating options
                        rating_options = ["1 out of 5", "2 out of 5", "3 out of 5", "4 out of 5", "5 out of 5"]
                        radio_groups.append(f"- Skill rating, {radio}, ref:{ref}, options:[{','.join(rating_options)}]")
                    elif "?" in radio:
                        # This is a yes/no question
                        radio_groups.append(f"- Yes/No question, {radio}, ref:{ref}, options:[Yes,No]")
                    else:
                        radio_groups.append(f"- Radio group, {radio}, ref:{ref}")
                    added_refs.add(ref)
            
            # 4. Process checkboxes
            checkboxes_list = self.state.page_state.get("checkboxes", [])
            for checkbox in checkboxes_list:
                ref = refs.get(checkbox, [''])[0]
                if ref and ref not in added_refs:
                    checkboxes.append(f"- Checkbox, {checkbox}, ref:{ref}, checked:false")
                    added_refs.add(ref)

            # 5. Categorize buttons
            buttons_list = self.state.page_state.get("buttons", [])
            submission_keywords = ["submit", "next", "continue", "apply", "finish"]
            interactive_keywords = ["add", "more", "upload", "browse", "choose", "select", "date"]
            
            for button in buttons_list:
                ref = refs.get(button, [''])[0]
                if ref and ref not in added_refs:
                    button_lower = button.lower()
                    if any(keyword in button_lower for keyword in submission_keywords):
                        submission_buttons.append(f"- Submit button, '{button}', ref:{ref}")
                    elif any(keyword in button_lower for keyword in interactive_keywords):
                        interactive_buttons.append(f"- Action button, '{button}', ref:{ref}")
                    else:
                        interactive_buttons.append(f"- Button, '{button}', ref:{ref}")
                    added_refs.add(ref)
        
        # Combine all fillable fields in a logical order
        all_fillable = []
        
        # Add form fields in order of appearance if possible
        if self.state.page_state and "snapshot_text" in self.state.page_state:
            snapshot_lines = self.state.page_state["snapshot_text"].splitlines()
            
            # Track which elements we've already added
            added_elements = set()
            
            # Process each line to maintain original order
            for line in snapshot_lines:
                # Check for text fields
                for field in form_fields:
                    field_label = field.split(',', 1)[1].strip().split(',')[0].strip()
                    if field_label in line and field not in added_elements:
                        all_fillable.append(field)
                        added_elements.add(field)
                        break
                
                # Check for dropdowns
                for dropdown in dropdowns:
                    dropdown_label = dropdown.split(',', 1)[1].strip().split(',')[0].strip()
                    if dropdown_label in line and dropdown not in added_elements:
                        all_fillable.append(dropdown)
                        added_elements.add(dropdown)
                        break
                        
                # Check for radio groups
                for radio in radio_groups:
                    radio_label = radio.split(',', 1)[1].strip().split(',')[0].strip()
                    if radio_label in line and radio not in added_elements:
                        all_fillable.append(radio)
                        added_elements.add(radio)
                        break
                        
                # Check for checkboxes
                for checkbox in checkboxes:
                    checkbox_label = checkbox.split(',', 1)[1].strip().split(',')[0].strip()
                    if checkbox_label in line and checkbox not in added_elements:
                        all_fillable.append(checkbox)
                        added_elements.add(checkbox)
                        break
                        
                # Check for interactive buttons that might reveal form fields
                for button in interactive_buttons:
                    button_label = button.split(',', 1)[1].strip().split(',')[0].strip().strip("'")
                    if button_label in line and button not in added_elements:
                        all_fillable.append(button)
                        added_elements.add(button)
                        break
        
        # If we couldn't determine order from snapshot, just combine all lists
        if not all_fillable:
            all_fillable = form_fields + dropdowns + radio_groups + checkboxes + interactive_buttons
        
        # Use the enhanced form field parser to get clean output
        form_field_text = "\n".join(all_fillable)
        clean_formatted_fields = parse_form_fields_enhanced(form_field_text)
        
        # Build the context parts
        context_parts = [
            f"=== CURRENT PAGE ===\n{self.state.current_url}",
            f"\n=== FILL_FIELDS ===\n{clean_formatted_fields}"
        ]
        
        # Add non-fillable buttons separately
        if submission_buttons:
            context_parts.append("\n=== SUBMISSION BUTTONS ===\n" + "\n".join(submission_buttons))
        
        # Add file upload areas if found
        file_uploads = self.state.page_state.get("file_uploads", [])
        if file_uploads:
            file_upload_text = []
            for file_upload in file_uploads:
                # Try to find ref for file upload
                upload_ref = ""
                for label, ref_list in refs.items():
                    if "browse files" in label.lower() or "upload" in label.lower():
                        upload_ref = ref_list[0] if ref_list else ""
                        break
                file_upload_text.append(f"- File upload, {file_upload}, ref:{upload_ref}")
            context_parts.append("\n=== FILE UPLOADS ===\n" + "\n".join(file_upload_text))
        
        page_context = "\n".join(context_parts)
        
        # Debug: Print the page context being sent to the LLM
        print("\n" + "="*80)
        print("PAGE CONTEXT BEING SENT TO LLM:")
        print("="*80)
        print(page_context)
        print("="*80 + "\n")
        
        filler_prompt = FILLER_PROMPT_TEMPLATE.format(page_context=page_context)
        
        # Create a proper message sequence for the LLM
        filler_messages = [
            SystemMessage(content=filler_prompt),
            HumanMessage(content="Please fill out this form with appropriate dummy data."),
        ]

        try:
            response = self.llm.invoke(filler_messages)
            if not response.content:
                response.content = "Planning to fill out the form fields."
            return {"messages": messages + [response]}
        except Exception as e:
            error_msg = f"Error in filler node: {str(e)}"
            return {"messages": messages + [AIMessage(content=error_msg)]}

    def route_after_execution(self, state: MessagesState):
        """Route after tool execution back to planner."""
        return "planner"


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