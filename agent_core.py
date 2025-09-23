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
from utils import extract_interactive_elements, truncate_text, find_element_ref
from prompts import SYSTEM_PROMPT_TEMPLATE, FILLER_PROMPT_TEMPLATE
from tree_parser import parse_snapshot  # Import the new tree parser

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
    max_tool_time: int = 30
    errors: List[str] = field(default_factory=list)
    task_complete: bool = False
 
    @property
    def last_message(self):
        return self.messages[-1] if self.messages else None


class WebAgent:
    """Intelligent web automation agent."""
    
    def __init__(self, llm, tool_manager: ToolManager, user_data: Dict = None):
        self.llm = llm
        self.tool_manager = tool_manager
        self.state = AgentState()
        self.user_data = user_data or {}

    def _find_matching_user_data(self, field_label: str) -> Optional[str]:
        """Find matching user data for a given field label using fuzzy matching."""
        if not field_label or not self.user_data:
            return None
        
        # Normalize the field label
        field_lower = field_label.lower().strip()
        
        # Remove common prefixes and suffixes
        for prefix in ['your_', 'enter_', 'please_', 'input_']:
            if field_lower.startswith(prefix):
                field_lower = field_lower[len(prefix):]
        
        for suffix in ['_here', '_field', '_input']:
            if field_lower.endswith(suffix):
                field_lower = field_lower[:-len(suffix)]
        
        # Clean up special characters and spaces
        field_clean = field_lower.replace('_', '').replace('-', '').replace(' ', '').replace('*', '')
        
        # Look for exact matches or partial matches
        for field_keys, value in self.user_data.items():
            if isinstance(field_keys, tuple):
                for key in field_keys:
                    key_clean = key.lower().replace('_', '').replace('-', '').replace(' ', '')
                    if key_clean == field_clean or key_clean in field_clean or field_clean in key_clean:
                        return value
            else:
                # Handle legacy format
                key_clean = field_keys.lower().replace('_', '').replace('-', '').replace(' ', '')
                if key_clean == field_clean or key_clean in field_clean or field_clean in key_clean:
                    return value
        
        return None

    def _build_user_data_context(self) -> str:
        """Build a readable context string from user data for the LLM."""
        if not self.user_data:
            return "No user data available. Skip fields that cannot be filled."
        
        context_lines = ["=== AVAILABLE USER DATA ==="]
        
        for field_keys, value in self.user_data.items():
            if isinstance(field_keys, tuple):
                primary_key = field_keys[0]
                alternatives = ', '.join(field_keys[1:3])  # Show first few alternatives
                context_lines.append(f"- {primary_key} (also matches: {alternatives}): '{value}'")
            else:
                context_lines.append(f"- {field_keys}: '{value}'")
        
        context_lines.extend([
            "",
            "INSTRUCTIONS FOR FIELD MATCHING:",
            "- Match form field labels to the most appropriate user data above",
            "- Use fuzzy matching (ignore case, underscores, spaces, special characters)",
            "- For file uploads, use the provided file paths",
            "- If no matching data exists for a field, skip that field unless it's clearly required",
            "- For required fields without data, use reasonable defaults or leave empty"
        ])
        
        return "\n".join(context_lines)

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

                # Normalize select option arguments: tool expects 'values' as an array
                if tool_name == "browser_select_option":
                    if not tool_args.get("values"):
                        # Accept common variants and coerce to list
                        for k in ["text", "value", "option", "label"]:
                            if k in tool_args and tool_args[k]:
                                v = tool_args[k]
                                tool_args["values"] = v if isinstance(v, list) else [v]
                                break
                    # Remove non-schema keys that might confuse the tool
                    for k in ["text", "value", "option", "label"]:
                        if k in tool_args:
                            tool_args.pop(k, None)
                
                # Print raw metadata before execution
                print(f"\n[TOOL_METADATA] name={tool_name} args={tool_args} call_id={tool_id}")

                # Execute tool with timeout and single retry on transient errors
                async def _run_once():
                    return await self.tool_manager.execute_tool(tool_name, tool_args)

                attempt = 0
                max_attempts = 2 if tool_name in ["browser_click", "browser_select_option", "browser_type"] else 1
                last_error: Optional[Exception] = None
                result = None
                while attempt < max_attempts:
                    attempt += 1
                    try:
                        result = await asyncio.wait_for(_run_once(), timeout=getattr(self.state, "max_tool_time", 30))
                        print(f"   - {tool_name} executed successfully (attempt {attempt}/{max_attempts})")
                        break
                    except asyncio.TimeoutError as te:
                        last_error = te
                        print(f"   - Timeout executing {tool_name} (attempt {attempt}/{max_attempts}) after {getattr(self.state, 'max_tool_time', 30)}s")
                    except Exception as e:
                        last_error = e
                        print(f"   - Error executing {tool_name} (attempt {attempt}/{max_attempts}): {str(e)}")

                    # Before retry, try to refresh snapshot to get latest refs/state
                    if attempt < max_attempts and "browser_snapshot" in getattr(self.tool_manager, "tool_schemas", {}):
                        try:
                            print("   - Refreshing snapshot before retry...")
                            await asyncio.wait_for(self.tool_manager.execute_tool("browser_snapshot", {}), timeout=20)
                        except Exception as e:
                            print(f"   - Snapshot refresh failed: {str(e)}")

                if result is None and last_error is not None:
                    raise last_error
                
                # Special handling for snapshot
                if tool_name == "browser_snapshot":
                    # Store the raw snapshot text for element finding
                    self.state.page_state["snapshot_text"] = result
                    page_elements = extract_interactive_elements(result)
                    self.state.page_state.update(page_elements)
                    
                    # NEW: Also parse using tree parser for form fields
                    try:
                        tree_form_fields = parse_snapshot(result)
                        self.state.page_state["tree_form_fields"] = tree_form_fields
                        print(f"   - Tree parser extracted {len(tree_form_fields)} field groups")
                    except Exception as e:
                        print(f"   - Tree parser error: {str(e)}")
                        self.state.page_state["tree_form_fields"] = {}
                    
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

        # Enhanced form detection using tree parser results
        if self.state.page_state and "tree_form_fields" in self.state.page_state:
            tree_fields = self.state.page_state["tree_form_fields"]
            
            # Count different types of form fields
            form_indicators = 0
            total_fields = 0
            
            for field_group, fields in tree_fields.items():
                for field in fields:
                    total_fields += 1
                    field_type = field.get('type', '')
                    if field_type in ['textbox', 'combobox', 'checkbox', 'radio', 'spinbutton']:
                        form_indicators += 1
            
            # If we have significant form content, route to filler
            if total_fields >= 2 and form_indicators >= 1:
                # Check if we're not already in a filling loop
                recent_fill_attempts = len([m for m in messages[-3:] if isinstance(m, ToolMessage) and "browser_type" in str(m.content)])
                if recent_fill_attempts < 3:  # Allow up to 3 consecutive fill attempts
                    print(f"   - Routing to filler: {total_fields} total fields, {form_indicators} form fields")
                    return "filler"
        
        # Fallback to original detection method if tree parser didn't find enough
        if self.state.page_state:
            form_indicators = 0
            
            # Check for multiple input fields
            inputs = self.state.page_state.get("inputs", [])
            if len(inputs) >= 2:  # At least two input fields
                form_indicators += 1
            
            # Check for common form field names
            common_fields = {"name", "email", "phone", "address", "password"}
            field_matches = 0
            for field in inputs:
                if not isinstance(field, str):
                    continue
                f_lower = field.lower()
                if any(common in f_lower for common in common_fields):
                    field_matches += 1
            if field_matches >= 2:  # Increased from 1 to 2
                form_indicators += 1
            
            # Check for required fields (marked with *)
            required_fields = sum(1 for field in inputs if isinstance(field, str) and "*" in field)
            if required_fields > 0:
                form_indicators += 1
                
            # Check for submit/continue buttons along with inputs
            buttons = self.state.page_state.get("buttons", [])
            submit_buttons = any(isinstance(b, str) and any(k in b.lower() for k in ["submit", "continue", "next", "apply"]) for b in buttons)
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
        """Specialized node for filling forms using user data and tree-parsed form fields."""
        messages = state["messages"]
        
        # Build user data context for the LLM
        user_data_context = self._build_user_data_context()
        
        # Use tree-parsed form fields if available
        if self.state.page_state and "tree_form_fields" in self.state.page_state:
            tree_form_fields = self.state.page_state["tree_form_fields"]
            
            # Build structured form context directly from tree parser results
            form_context_parts = [
                f"=== CURRENT PAGE ===\n{self.state.current_url}",
                "\n=== PARSED FORM FIELDS (Tree Structure) ==="
            ]
            
            # Convert tree fields to a structured format for the LLM
            field_descriptions = []
            submission_buttons = []
            
            for field_group_name, fields in tree_form_fields.items():
                if not fields:  # Skip empty field groups
                    continue
                    
                # Add field group header if it's not a generic name
                if field_group_name not in ['textbox', 'button', 'combobox', 'checkbox', 'radio']:
                    field_descriptions.append(f"\n--- {field_group_name} ---")
                
                for field in fields:
                    field_type = field.get('type', 'unknown')
                    ref = field.get('ref', 'no-ref')
                    
                    if field_type == 'textbox':
                        placeholder = field.get('placeholder', '')
                        placeholder_text = f", placeholder: '{placeholder}'" if placeholder else ""
                        field_descriptions.append(f"- Text field '{field_group_name}', ref: {ref}, type: text{placeholder_text}")
                    
                    elif field_type == 'combobox':
                        options = field.get('options', [])
                        selected = field.get('selected', '')
                        options_text = f", options: [{', '.join(options)}]" if options else ""
                        selected_text = f", selected: '{selected}'" if selected else ""
                        field_descriptions.append(f"- Dropdown '{field_group_name}', ref: {ref}, type: select{options_text}{selected_text}")
                    
                    elif field_type == 'radio':
                        label = field.get('label', '')
                        checked = field.get('checked', False)
                        field_descriptions.append(f"- Radio option '{label}', ref: {ref}, type: radio, checked: {checked}")
                    
                    elif field_type == 'checkbox':
                        checked = field.get('checked', False)
                        field_descriptions.append(f"- Checkbox '{field_group_name}', ref: {ref}, type: checkbox, checked: {checked}")
                    
                    elif field_type == 'spinbutton':
                        label = field.get('label', '')
                        field_descriptions.append(f"- Number field '{field_group_name}', ref: {ref}, type: number")
                    
                    elif field_type == 'button':
                        label = field.get('label', field_group_name)
                        # Guard against non-string labels
                        label_str = label if isinstance(label, str) else str(label)
                        if any(keyword in label_str.lower() for keyword in ['submit', 'send', 'apply', 'continue', 'next', 'finish']):
                            submission_buttons.append(f"- Submit button '{label}', ref: {ref}, type: button")
                        else:
                            field_descriptions.append(f"- Button '{label}', ref: {ref}, type: button")
            
            # Add the structured fields to context
            if field_descriptions:
                form_context_parts.append("\n".join(field_descriptions))
            
            # Add submission buttons separately
            if submission_buttons:
                form_context_parts.append("\n=== SUBMISSION BUTTONS ===")
                form_context_parts.append("\n".join(submission_buttons))
            
            # Add file upload information if needed
            # file_uploads = self.state.page_state.get("file_uploads", [])
            # if file_uploads:
            #     form_context_parts.append("\n=== FILE UPLOADS (USE browser_file_upload TOOL) ===")
            #     form_context_parts.append("File path: /Users/arhan/Desktop/clean-connection 2/sample.pdf")
            #     for upload in file_uploads:
            #         form_context_parts.append(f"- {upload}")
            
            page_context = "\n".join(form_context_parts)
            
            # Debug: Print the page context being sent to the LLM
            print("\n" + "="*80)
            print("TREE-PARSED FORM CONTEXT BEING SENT TO LLM:")
            print("="*80)
            print(page_context)
            print("="*80 + "\n")
        
        else:
            # Fallback to original form field extraction if tree parser failed
            print("   - Tree form fields not available, using fallback method")
            
            # [Keep original form field extraction code as fallback]
            form_fields = []
            interactive_buttons = []
            submission_buttons = []
            dropdowns = []
            radio_groups = []
            checkboxes = []
            file_uploads = []
            
            if self.state.page_state:
                refs = self.state.page_state.get("refs", {})
                buttons = self.state.page_state.get("buttons", [])
                inputs = self.state.page_state.get("inputs", [])
                comboboxes = self.state.page_state.get("comboboxes", [])
                radio_labels = self.state.page_state.get("radio_groups", [])
                checkbox_labels = self.state.page_state.get("checkboxes", [])
                uploads = self.state.page_state.get("file_uploads", [])

                # Text inputs
                for label in inputs:
                    label_refs = refs.get(label, [])
                    if not label_refs:
                        # Try generic mapping: sometimes labels may be missing; skip if no ref
                        continue
                    ref = label_refs[0]
                    form_fields.append(f"- Text field '{label}', ref: {ref}, type: text")

                # Dropdowns / Comboboxes
                for label in comboboxes:
                    label_refs = refs.get(label, [])
                    # support combobox_<ref> synthetic ids
                    if not label_refs and label.startswith("combobox_"):
                        label_refs = refs.get(label, [])
                    if not label_refs:
                        continue
                    ref = label_refs[0]
                    # try to find options if stored
                    options_key = f"{label}_options"
                    options = refs.get(options_key, [])
                    options_text = f", options: [{', '.join(options)}]" if options else ""
                    dropdowns.append(f"- Dropdown '{label}', ref: {ref}, type: select{options_text}")

                # Radio groups/options
                for label in radio_labels:
                    label_refs = refs.get(label, [])
                    if not label_refs:
                        continue
                    ref = label_refs[0]
                    radio_groups.append(f"- Radio group '{label}', ref: {ref}")

                # Checkboxes
                for label in checkbox_labels:
                    label_refs = refs.get(label, [])
                    if not label_refs:
                        continue
                    ref = label_refs[0]
                    checkboxes.append(f"- Checkbox '{label}', ref: {ref}, type: checkbox")

                # Buttons
                for label in buttons:
                    label_refs = refs.get(label, [])
                    ref_str = f", ref: {label_refs[0]}" if label_refs else ""
                    item = f"- Button '{label}'{ref_str}, type: button"
                    # Guard against non-string labels
                    label_str = label if isinstance(label, str) else str(label)
                    if any(k in label_str.lower() for k in ['submit','send','apply','continue','next','finish']):
                        submission_buttons.append(item)
                    else:
                        interactive_buttons.append(item)

                # File uploads
                for upload in uploads:
                    # uploads may be synthetic like file_upload_e12 with refs mapping
                    label_refs = refs.get(upload, [])
                    if not label_refs:
                        continue
                    ref = label_refs[0]
                    file_uploads.append(f"- File upload '{upload}', ref: {ref}")
            
            # Build context from fallback extraction
            all_fillable = form_fields + dropdowns + radio_groups + checkboxes + interactive_buttons + file_uploads
            form_field_text = "\n".join(all_fillable) if all_fillable else "No form fields detected"
            
            context_parts = [
                f"=== CURRENT PAGE ===\n{self.state.current_url}",
                f"\n=== FILLABLE FIELDS (Fallback) ===\n{form_field_text}"
            ]
            
            if submission_buttons:
                context_parts.append("\n=== SUBMISSION BUTTONS ===\n" + "\n".join(submission_buttons))
            
            page_context = "\n".join(context_parts)
        
        # Combine user data context with page context
        combined_context = f"{user_data_context}\n\n{page_context}"
        
        # Use the filler prompt template
        filler_prompt = FILLER_PROMPT_TEMPLATE.format(page_context=combined_context)
        
        # Create a proper message sequence for the LLM
        filler_messages = [
            SystemMessage(content=filler_prompt),
            HumanMessage(content="Please fill out this form by matching field labels to the available user data. Use the ref values provided to target the correct elements. Skip fields that don't have matching user data unless they're clearly required."),
        ]

        try:
            response = self.llm.invoke(filler_messages)
            if not response.content:
                response.content = "Planning to fill out the form fields using available user data."
            return {"messages": messages + [response]}
        except Exception as e:
            error_msg = f"Error in filler node: {str(e)}"
            return {"messages": messages + [AIMessage(content=error_msg)]}


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