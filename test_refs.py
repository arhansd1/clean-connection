# quick test for find_element_ref behavior
from utils import find_element_ref

snap = r'''
- radiogroup "Are you willing to relocate?" [ref=e101]
- option "Yes" [ref=e102]
- option "No" ref=e103
- checkbox "Agree terms" ref: e104
- button "Submit" [ref=e200]
'''

job = r'''
- checkboxes "weekly availability days and hours daily" [ref=e105]
- option "monday"c [ref=e106]
- option "tuesday" [ref=e107]
- option "wednesday" [ref=e108]
- option "thrusday" [ref=109]
- option "friday" [ref=110]
- option "saturday" [ref=111]
- option "sunday(weekly off)" [ref=112]
'''
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

            
            if not raw_fields: ## if raw filed is empty then use invoke_messages
                response = self.llm.invoke(invoke_messages)
            else: # if raw field is not empty, there are some fields in raw_fields then use final_messages
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
    

print(find_element_ref(job, "weekly availability day and hours daily", element_type="checkboxes"))
print(find_element_ref(snap, "Are you willing to relocate", element_type="radio"))  # expect e101
print(find_element_ref(snap, "Yes", element_type="radio"))                          # expect e102 or e103 if present
print(find_element_ref(snap, "Agree terms", element_type="checkbox"))               # expect e104


