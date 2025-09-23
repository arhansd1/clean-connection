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
from utils import extract_interactive_elements, truncate_text, find_element_ref, parse_form_fields_enhanced
from prompts import SYSTEM_PROMPT_TEMPLATE, FILLER_PROMPT_TEMPLATE

import base64
import os
from pathlib import Path
import re

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
########
        raw_fields = self.state.page_state.get("raw_fields", {})
        #print("raw_fields:", json.dumps(raw_fields, indent=2))

        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
            page_context=raw_fields,
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
            raw_fields = {}
            if self.state.page_state:
                # Add page state into invoke_messages for context
                raw_fields = self.state.page_state.get("raw_fields", {})

                # human message with page context
            final_messages = invoke_messages + [HumanMessage(content=f"=== PAGE CONTEXT ===\n{raw_fields}")]

            # Call LLM with structured messages

            
            if not raw_fields:
                response = self.llm.invoke(invoke_messages)
            else:
                response = self.llm.invoke(final_messages)


            

            # If LLM returned tool calls but no content, add descriptive content
            if hasattr(response, "tool_calls") and response.tool_calls and not getattr(response, "content", None):
                tools_desc = "; ".join([f"{tc.get('name')}({tc.get('args')})" for tc in response.tool_calls])
                response.content = f"Planning next steps: {tools_desc}"

            self.state.step_count += 1
            #print(response)
            # Add both the rebuilt system prompt and response to history
            return {"messages": messages + [response]}

        except Exception as e:
            error_msg = f"Error in planning: {str(e)}"
            self.state.errors.append(error_msg)
            return {"messages": messages + [AIMessage(content=error_msg)]}      
    
    async def executor_node(self, state: MessagesState):
        """Execute tools called by the planner."""

        #Extracts the last message (which should contain tool calls)
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
                if self.state.page_state and "snapshot_text" in self.state.page_state:
                    element_text = (
                        tool_args.get("element")
                        or tool_args.get("selector", "")
                        or tool_args.get("ref", "")
                    )

                    # If we only have a ref, try to find its text from page state
                    if str(element_text).startswith("e") and "refs" in self.state.page_state:
                        for label, refs in self.state.page_state.get("refs", {}).items():
                            if element_text in refs:
                                element_text = label
                                break

                    # Try to find ref
                    new_ref = find_element_ref(
                        self.state.page_state["snapshot_text"],
                        element_text,
                        tool_args.get("element_type", "button"),
                    )

                    if new_ref:
                        tool_args["ref"] = new_ref
                        tool_args["element"] = element_text

                        # 🔎 Detect checkboxes or radios
                        checkboxes = self.state.page_state.get("checkboxes", [])
                        radios = self.state.page_state.get("radio_groups", [])

                        is_checkbox = any(new_ref in self.state.page_state["refs"].get(cb, []) for cb in checkboxes)
                        is_radio = any(new_ref in self.state.page_state["refs"].get(rg, []) for rg in radios)

                        if is_checkbox or is_radio:
                            tool_args["force_check"] = True  # mark for Playwright `.check()` instead of `.click()`


            
            # # Handle click operations
            # if tool_name == "browser_click":
            #     # Always try to find the element in the latest snapshot
            #     if self.state.page_state and "snapshot_text" in self.state.page_state:
            #         # Get element text, trying different possible keys
            #         element_text = tool_args.get("element") or tool_args.get("selector", "") or tool_args.get("ref", "")
                    
            #         # If we only have a ref, try to find its text from page state
            #         if element_text.startswith("e") and "refs" in self.state.page_state:
            #             for label, refs in self.state.page_state.get("refs", {}).items():
            #                 if element_text in refs:
            #                     element_text = label
            #                     break
                    
            #         # Try to find ref
            #         new_ref = find_element_ref(
            #             self.state.page_state["snapshot_text"], 
            #             element_text,
            #             tool_args.get("element_type", "button")
            #         )
                    
            #         if new_ref:
            #             # Use both element and ref for better reliability
            #             tool_args["ref"] = new_ref
            #             tool_args["element"] = element_text


                    # Handle Yes/No with context
                    # if element_text.lower() in ["yes", "no"]:
                    #     question_text = tool_args.get("question")  # <- add question context if available
                    #     refs_dict = self.state.page_state.get("refs", {})

                    #     for q_text, options in refs_dict.items():
                    #         if question_text and question_text.lower() in q_text.lower():
                    #             for opt in options:
                    #                 if opt["label"].lower() == element_text.lower():
                    #                     tool_args["ref"] = opt["ref"]
                    #                     tool_args["element"] = element_text
                    #                     break
                    #             break

                    # else:
                    #     # Fallback for other buttons
                    #     new_ref = find_element_ref(
                    #         self.state.page_state["snapshot_text"],
                    #         element_text,
                    #         tool_args.get("element_type", "button")
                    #     )
                    #     if new_ref:
                    #         tool_args["ref"] = new_ref
                    #         tool_args["element"] = element_text


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


                            
                if tool_name == "browser_type":
                    if "value" in tool_args and "text" not in tool_args:
                        tool_args["text"] = tool_args.pop("value")

                # Ensure both element and ref are present for relevant tools
                if tool_name in ["browser_type", "browser_click", "browser_select_option"]:
                    # If we have ref but no element, try to find the element from page state
                    if "ref" in tool_args and "element" not in tool_args:
                        if self.state.page_state and "refs" in self.state.page_state:
                            for element, refs in self.state.page_state["refs"].items():
                                if tool_args["ref"] in refs:
                                    tool_args["element"] = element
                                    break

                                # Normalize args for browser_fill_form to the expected shape
                if tool_name == "browser_fill_form":
                    if isinstance(tool_args, dict) and "fields" not in tool_args:
                        #tool_args = {"fields": [{"name": k, "value": v} for k, v in tool_args.items()]}
                        tool_args = {k: v for k, v in tool_args.items()}
            

                # Print raw metadata before execution
                #print(f"\n[TOOL_METADATA] name={tool_name} args={tool_args} call_id={tool_id}")
                
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
                    print(summary)
                    results.append(ToolMessage(content=page_elements, tool_call_id=tool_id))
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
    
    
    # for tool_call in getattr(last_message, "tool_calls", []) or []:
    #     tool_name = tool_call.get("name")
    #     tool_args = tool_call.get("args",[])
               


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
        file_uploads = []
        
        if self.state.page_state:
            # Get all element references
            refs = self.state.page_state.get("refs", {})
            
            # Track which refs we've already added to avoid duplicates
            added_refs = set()
            
            # 1. Process regular input fields
            inputs = self.state.page_state.get("inputs", [])
            for input_field in inputs:
                field_refs = refs.get(input_field, [])
                for ref in field_refs:
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
                            
                        # form_fields.append({
                        #     "label": input_field,
                        #     "ref": ref,
                        #     "type": field_type,   # e.g. "textbox", "checkbox"
                        #     "value": None         # placeholder, fill later
                        # })
                        # added_refs.add(ref)

                        form_fields.append(f"- Text field, {input_field}, ref:{ref}, type:{field_type}, value:''")
                        added_refs.add(ref)

            
            # 2. Process dropdowns/selects with their options - USE REF-BASED IDENTIFICATION
            dropdowns_list = self.state.page_state.get("comboboxes", [])
            for dropdown_id in dropdowns_list:
                if dropdown_id.startswith("combobox_"):
                    ref = dropdown_id.replace("combobox_", "")
                    if ref and ref not in added_refs:
                        # Get options from refs storage
                        options_key = f"{dropdown_id}_options"
                        options = refs.get(options_key, [])
                        
                        options_str = f", options:[{','.join(options)}]" if options else ""
                        dropdowns.append(f"- Dropdown, ref:{ref}{options_str}, type:select")
                        added_refs.add(ref)
            #### ####
            # 3. Process radio groups and radio buttons
            radio_groups_list = self.state.page_state.get("radio_groups", [])
            for radio in radio_groups_list:
                radio_refs = refs.get(radio, [])
                for ref in radio_refs:
                    if ref and ref not in added_refs:
                        # Check if this is a skill rating or yes/no question
                        radio_lower = radio.lower()
                        if any(skill in radio_lower for skill in ["microsoft", "communication", "seo", "skill"]):
                            # This is a skill rating, find the rating options
                            rating_options = ["1 out of 5", "2 out of 5", "3 out of 5", "4 out of 5", "5 out of 5"]
                            radio_groups.append(f"- Skill rating, ref:{ref}, options:[{','.join(rating_options)}], type:radio")
                        elif "?" in radio:
                            # This is a yes/no question
                            radio_groups.append(f"- Yes/No question, {radio}, ref:{ref}, options:[Yes,No], type:radio")
                        else:
                            radio_groups.append(f"- Radio group, {radio}, ref:{ref}, type:radio")
                        added_refs.add(ref)
            
            # 4. Process checkboxes
            checkboxes_list = self.state.page_state.get("checkboxes", [])
            for checkbox in checkboxes_list:
                checkbox_refs = refs.get(checkbox, [])
                for ref in checkbox_refs:
                    if ref and ref not in added_refs:
                        checkboxes.append(f"- Checkbox, {checkbox}, ref:{ref}, checked:false, type:checkbox")
                        added_refs.add(ref)

            # 5. Categorize buttons
            buttons_list = self.state.page_state.get("buttons", [])
            submission_keywords = ["submit", "next", "continue", "apply", "finish"]
            interactive_keywords = ["add", "more", "upload", "browse", "choose", "select", "date"]
            
            for button in buttons_list:
                button_refs = refs.get(button, [])
                for ref in button_refs:
                    if ref and ref not in added_refs:
                        button_lower = button.lower()
                        if any(keyword in button_lower for keyword in submission_keywords):
                            submission_buttons.append(f"- Submit button, '{button}', ref:{ref}, type:button")
                        elif any(keyword in button_lower for keyword in interactive_keywords):
                            interactive_buttons.append(f"- Action button, '{button}', ref:{ref}, type:button")
                        else:
                            interactive_buttons.append(f"- Button, '{button}', ref:{ref}, type:button")
                        added_refs.add(ref)
        
        # 6. Process file uploads - ENHANCED WITH CLEAR LABELS AND TYPE
        file_upload_list = self.state.page_state.get("file_uploads", [])
        for file_upload_id in file_upload_list:
            if file_upload_id.startswith("file_upload_"):
                ref = file_upload_id.replace("file_upload_", "")
                if ref and ref not in added_refs:
                    # Try to find a descriptive label for this file upload
                    file_label = "File Upload"
                    # Look for labels that contain upload-related keywords
                    for label, ref_list in refs.items():
                        if ref in ref_list and any(kw in label.lower() for kw in ["upload", "file", "cv", "resume", "cover", "browse"]):
                            file_label = label
                            break
                    
                    # Also check the original file uploads list for context
                    original_uploads = self.state.page_state.get("file_uploads_detailed", [])
                    for upload in original_uploads:
                        if f"ref:{ref}" in upload:
                            # Extract label from the original upload entry
                            label_match = re.search(r'file_upload: (.*?)(?:,|$)', upload)
                            if label_match:
                                file_label = label_match.group(1)
                    
                    file_uploads.append(f"- File upload, {file_label}, ref:{ref}, type:file")
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
                    ref = field["ref"] if isinstance(field, dict) else field.split('ref:')[-1].split(',')[0].strip()
                    if f"ref:{ref}" in line and field not in added_elements:
                        all_fillable.append(field)
                        added_elements.add(field)
                        break

                # Check for dropdowns
                for dropdown in dropdowns:
                    ref = dropdown["ref"] if isinstance(dropdown, dict) else dropdown.split('ref:')[-1].split(',')[0].strip()
                    if f"ref:{ref}" in line and dropdown not in added_elements:
                        all_fillable.append(dropdown)
                        added_elements.add(dropdown)
                        break

                # Check for radio groups
                for radio in radio_groups:
                    ref = radio["ref"] if isinstance(radio, dict) else radio.split('ref:')[-1].split(',')[0].strip()
                    if f"ref:{ref}" in line and radio not in added_elements:
                        all_fillable.append(radio)
                        added_elements.add(radio)
                        break

                # Check for checkboxes
                for checkbox in checkboxes:
                    ref = checkbox["ref"] if isinstance(checkbox, dict) else checkbox.split('ref:')[-1].split(',')[0].strip()
                    if f"ref:{ref}" in line and checkbox not in added_elements:
                        all_fillable.append(checkbox)
                        added_elements.add(checkbox)
                        break

                # Check for interactive buttons that might reveal form fields
                for button in interactive_buttons:
                    ref = button["ref"] if isinstance(button, dict) else button.split('ref:')[-1].split(',')[0].strip()
                    if f"ref:{ref}" in line and button not in added_elements:
                        all_fillable.append(button)
                        added_elements.add(button)
                        break

                # Check for file uploads
                for file_upload in file_uploads:
                    ref = file_upload["ref"] if isinstance(file_upload, dict) else file_upload.split('ref:')[-1].split(',')[0].strip()
                    if f"ref:{ref}" in line and file_upload not in added_elements:
                        all_fillable.append(file_upload)
                        added_elements.add(file_upload)
                        break
        
        # If we couldn't determine order from snapshot, just combine all lists
        if not all_fillable:
            all_fillable = form_fields + dropdowns + radio_groups + checkboxes + interactive_buttons + file_uploads
        
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
        
        # Add file upload areas if found - CLEARLY MARKED AS FILE TYPE
        if file_uploads:
            context_parts.append("\n=== FILE UPLOADS (USE browser_file_upload TOOL) ===")
            context_parts.append("File path: /Users/dineshk/Downloads/clean-connection-2/sample.pdf")
            context_parts.append("\n".join(file_uploads))
        
        page_context = "\n".join(context_parts)
        
        raw_fields = self.state.page_state.get("raw_fields", {})
        #print("raw_fields:", json.dumps(raw_fields, indent=2))

        # Debug: Print the page context being sent to the LLM
        print("\n" + "="*80)
        print("PAGE CONTEXT BEING SENT TO LLM:")
        print("="*80)
        print(raw_fields)
        print("="*80 + "\n")
        
        # Use the updated prompt template with explicit file upload instructions
        filler_prompt = FILLER_PROMPT_TEMPLATE.format(page_context=raw_fields)
        
        # Create a proper message sequence for the LLM
        filler_messages = [
            SystemMessage(content=filler_prompt),
            HumanMessage(content="Please fill out this form with appropriate dummy data. Remember to use browser_file_upload for any file upload fields!"),
        ]

        try:
            response = self.llm.invoke(filler_messages)
            if not response.content:
                response.content = "Planning to fill out the form fields including file uploads."

            # 🔹 If the LLM returned a plain dict of label->value, wrap it
            #     into the shape the tool expects later (best-effort).

            if isinstance(response.content, dict) and "fields" not in response.content:
                structured_fields = [{"name": k, "value": v} for k, v in response.content.items()]
                response.content = {"fields": structured_fields}


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