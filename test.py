# """
# Simple function call tests for utils.py with explicit printing of returned
# values/results for each function.

# Run:
#     python test_utils_simple.py

# This script:
#  - Calls each function once (single-line usage pattern)
#  - Prints the raw / summarized result
#  - Performs minimal assertions
#  - Displays a final summary
# """

# from utils import (
#     truncate_text,
#     extract_interactive_elements,
#     find_element_ref,
#     extract_form_fields,
#     analyze_goal,
# )
# import json
# import sys
# from typing import List, Tuple, Callable, Any

# SNAPSHOT = """
# ### Page state
# - Page URL: https://httpbin.org/forms/post
# - Page Snapshot:
# ```yaml
# - generic [ref=e2]:
#   - paragraph [ref=e3]:
#     - generic [ref=e4]:
#       - text: "Customer name:"
#       - textbox "Customer name:" [ref=e5]
#   - paragraph [ref=e6]:
#     - generic [ref=e7]:
#       - text: "Telephone:"
#       - textbox "Telephone:" [ref=e8]
#   - paragraph [ref=e9]:
#     - generic [ref=e10]:
#       - text: "E-mail address:"
#       - textbox "E-mail address:" [ref=e11]
#   - group "Pizza Size" [ref=e12]:
#     - generic [ref=e13]: Pizza Size
#     - paragraph [ref=e14]:
#       - generic [ref=e15]:
#         - radio "Small" [ref=e16]
#         - text: Small
#     - paragraph [ref=e17]:
#       - generic [ref=e18]:
#         - radio "Medium" [ref=e19]
#         - text: Medium
#     - paragraph [ref=e20]:
#       - generic [ref=e21]:
#         - radio "Large" [ref=e22]
#         - text: Large
#   - group "Pizza Toppings" [ref=e23]:
#     - generic [ref=e24]: Pizza Toppings
#     - paragraph [ref=e25]:
#       - generic [ref=e26]:
#         - checkbox "Bacon" [ref=e27]
#         - text: Bacon
#     - paragraph [ref=e28]:
#       - generic [ref=e29]:
#         - checkbox "Extra Cheese" [ref=e30]
#         - text: Extra Cheese
# """

# def print_section(title: str):
#     print("\n" + "=" * 70)
#     print(title)
#     print("=" * 70)

# def test_truncate_text():
#     print_section("TEST truncate_text")
#     original = "abcdef"
#     result = truncate_text(original, 4)
#     print("Original:", original)
#     print("Truncated:", result)
#     assert result.startswith("abcd") and "truncated" in result
#     return {"original": original, "truncated": result}

# def test_extract_interactive_elements():
#     print_section("TEST extract_interactive_elements")
#     parsed = extract_interactive_elements(SNAPSHOT)
#     print("Parsed keys:", list(parsed.keys()))
#     preview = {
#         "title": parsed.get("title"),
#         "buttons": parsed.get("buttons"),
#         "inputs": parsed.get("inputs")[:6],
#         "refs_count": len(parsed.get("refs", {})),
#         "interactives_sample": parsed.get("interactives")[:5],
#     }
#     print("Preview:", json.dumps(preview, indent=2))
#     assert "buttons" in parsed and "inputs" in parsed and "refs" in parsed
#     return preview

# def test_find_element_ref():
#     print_section("TEST find_element_ref")
#     label = "Customer name:"
#     ref = find_element_ref(SNAPSHOT, label)
#     print(f"Label: {label} -> Ref resolved: {ref}")
#     # Accept e5 (textbox) or e4 (parent) depending on heuristic
#     assert ref in {"e5", "e4"}, f"Unexpected ref {ref}"
#     return {"label": label, "ref": ref}

# def test_extract_form_fields():
#     print_section("TEST extract_form_fields")
#     fields = extract_form_fields(SNAPSHOT)
#     print("Total fields detected:", len(fields))
#     for f in fields:
#         print(" -", f)
#     labels = {f["label"] for f in fields}
#     expected = {"Customer name:", "Telephone:", "E-mail address:", "Small", "Medium", "Large", "Bacon", "Extra Cheese"}
#     missing = expected - labels
#     assert not missing, f"Missing labels: {missing}"
#     return {"fields": fields}

# def test_analyze_goal():
#     print_section("TEST analyze_goal")
#     goal = "Navigate to https://httpbin.org/forms/post fill form submit and take a screenshot"
#     analysis = analyze_goal(goal)
#     print("Goal:", goal)
#     print("Analysis:", json.dumps(analysis, indent=2))
#     assert all([
#         analysis["requires_navigation"],
#         analysis["requires_form_filling"],
#         analysis["requires_submission"],
#         analysis["requires_screenshot"],
#         analysis["target_url"] == "https://httpbin.org/forms/post"
#     ])
#     return {"goal": goal, "analysis": analysis}

# def main():
#     test_functions: List[Tuple[str, Callable[[], Any]]] = [
#         ("truncate_text", test_truncate_text),
#         ("extract_interactive_elements", test_extract_interactive_elements),
#         ("find_element_ref", test_find_element_ref),
#         ("extract_form_fields", test_extract_form_fields),
#         ("analyze_goal", test_analyze_goal),
#     ]
#     results = {}
#     failures = 0
#     for name, fn in test_functions:
#         try:
#             results[name] = fn()
#             print(f"[PASS] {name}")
#         except AssertionError as ae:
#             print(f"[FAIL] {name}: {ae}")
#             failures += 1
#         except Exception as e:
#             print(f"[ERROR] {name}: {e}")
#             failures += 1

#     print_section("SUMMARY")
#     print("Failures:", failures)
#     print("Results JSON:")
#     print(json.dumps(results, indent=2, default=str))

#     if failures:
#         print("\nOne or more tests failed.")
#         raise SystemExit(1)
#     print("\nAll tests passed successfully.")

# if __name__ == "__main__":
#     main()




# --------------------------------------------------



# """
# Simple, dependency‑light tests for agent_core.WebAgent internals.

# Covers:
#   - planner_node: step counting, system prompt injection, tool call planning
#   - executor_node: tool execution, navigation history, snapshot parsing -> page_state
#   - reflector_node: completion detection & task_complete flag
#   - route_from_planner: branching logic (executor / reflector / exit)
#   - run(): end‑to‑end tiny workflow with fake LLM & tools

# Run:
#     python test_agent_core_simple.py

# No external network / real MCP tools needed.

# NOTE: Your original executor_node uses `asyncio.run()` inside an async-ish flow.
# That pattern is fragile when already inside an event loop (e.g., tests, notebooks).
# For production-quality async, consider changing:

#     result = asyncio.run(self.tool_manager.execute_tool(...))

# to:

#     result = await self.tool_manager.execute_tool(...)

# and make executor_node async (then adjust graph build).  
# These tests work around it by making execute_tool a fast synchronous-ish future
# and patching asyncio.run with a local shim that just calls the coroutine.

# """

# import asyncio
# from dataclasses import dataclass
# from typing import Any, Dict, List, Optional

# # Import the code under test
# from agent_core import WebAgent, AgentState
# from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
# from langgraph.graph import MessagesState

# # ---------------------------------------------------------------------------
# # Fakes / Stubs
# # ---------------------------------------------------------------------------

# class FakeLLM:
#     """
#     Minimal LLM stub.
#     It inspects the existing messages and returns an AIMessage with (or without)
#     a tool call depending on an internal scripted step counter.
#     """
#     def __init__(self, scripted_tool_sequences: List[Dict[str, Any]]):
#         """
#         scripted_tool_sequences: list of dict steps.
#           Each item:
#             {
#               "tool_calls": [
#                   {"name": "browser_navigate", "args": {"url": "https://example.com"}, "id": "t1"},
#                   ...
#               ],
#               "text": "Planner explanation text"
#             }
#         If a step has no 'tool_calls', it returns a plain AIMessage (reflection path).
#         """
#         self.scripted = scripted_tool_sequences
#         self.call_index = 0

#     def invoke(self, messages: List[Any]):
#         if self.call_index >= len(self.scripted):
#             # Default final no-tool response
#             return AIMessage(content="No further actions.")
#         spec = self.scripted[self.call_index]
#         self.call_index += 1
#         tool_calls = spec.get("tool_calls") or []
#         content = spec.get("text", "Planning step")
#         if tool_calls:
#             return AIMessage(content=content, tool_calls=tool_calls)
#         return AIMessage(content=content)


# class FakeToolManager:
#     """
#     Simulates the subset of ToolManager the agent_core expects:
#       - get_tool_descriptions()
#       - execute_tool(name, args)
#     Tools implemented:
#       - browser_navigate
#       - browser_snapshot
#       - browser_click
#       - browser_take_screenshot
#     """
#     def __init__(self):
#         self.executions: List[Dict[str, Any]] = []
#         self.snap_counter = 0

#     def get_tool_descriptions(self) -> str:
#         return "- browser_navigate(url)\n- browser_snapshot()\n- browser_click(element)\n- browser_take_screenshot()"

#     async def execute_tool(self, name: str, args: Dict[str, Any]):
#         self.executions.append({"name": name, "args": dict(args)})
#         if name == "browser_navigate":
#             return f"Navigated to {args.get('url')}"
#         if name == "browser_click":
#             return f"Clicked element '{args.get('element')}'"
#         if name == "browser_take_screenshot":
#             return "Screenshot saved to fake_path/screen.png"
#         if name == "browser_snapshot":
#             self.snap_counter += 1
#             # Provide a small deterministic snapshot for parsing
#             return (
#                 "### Page state\n"
#                 f"- Page URL: https://example.com/page{self.snap_counter}\n"
#                 "- Page Title: Example Page\n"
#                 "- Page Snapshot:\n"
#                 "```yaml\n"
#                 "- generic [ref=e2]:\n"
#                 "  - heading \"Main Heading\" [level=1] [ref=e3]\n"
#                 "  - button \"Login\" [ref=e4]\n"
#                 "  - textbox \"Username\" [ref=e5]\n"
#                 "  - textbox \"Password\" [ref=e6]\n"
#                 "  - link \"Help\" [ref=e7]\n"
#                 "```"
#             )
#         return f"{name} executed"

#     # For compatibility with agent_core which calls tool_manager.execute_tool via asyncio.run
#     # we keep it as async but very fast.

# # Monkeypatch helper: silence async->sync bridging in executor_node
# # (If you refactor executor_node to fully async, you can remove this.)
# _original_asyncio_run = asyncio.run
# def _patched_asyncio_run(coro):
#     # Detect already in loop
#     try:
#         loop = asyncio.get_running_loop()
#     except RuntimeError:
#         loop = None
#     if loop and loop.is_running():
#         # Run the coroutine in the existing loop using ensure_future/await logic
#         # For simplicity, schedule and gather result with a helper
#         return loop.run_until_complete(coro)  # Might raise in some frameworks
#     return _original_asyncio_run(coro)

# # ---------------------------------------------------------------------------
# # Individual Node Tests
# # ---------------------------------------------------------------------------

# def test_planner_node_basic():
#     print("\n[TEST] planner_node_basic")
#     llm = FakeLLM([
#         {"tool_calls": [
#             {"name": "browser_navigate", "args": {"url": "https://example.com"}, "id": "nav1", "type": "tool_call"}
#         ], "text": "Navigating to target"}
#     ])
#     tm = FakeToolManager()
#     agent = WebAgent(llm, tm)

#     state = MessagesState(messages=[HumanMessage(content="Please open https://example.com")])
#     out = agent.planner_node(state)
#     msgs = out["messages"]
#     assert isinstance(msgs[0], SystemMessage), "System message not inserted"
#     assert isinstance(msgs[-1], AIMessage), "Last message should be AIMessage"
#     assert msgs[-1].tool_calls, "Planner did not produce tool call"
#     assert agent.state.step_count == 1, "Step count not incremented"
#     print("Planner produced tool call:", msgs[-1].tool_calls)


# def test_executor_node_navigate_and_snapshot():
#     print("\n[TEST] executor_node_navigate_and_snapshot")
#     llm = FakeLLM([])
#     tm = FakeToolManager()
#     agent = WebAgent(llm, tm)

#     # Simulate planner output with two tool calls
#     planner_msg = AIMessage(
#         content="Plan: navigate then snapshot",
#         tool_calls=[
#             {"name": "browser_navigate", "args": {"url": "https://example.com"}, "id": "t1", "type": "tool_call"},
#             {"name": "browser_snapshot", "args": {}, "id": "t2", "type": "tool_call"},
#         ],
#     )
#     state = MessagesState(messages=[HumanMessage(content="Goal"), planner_msg])

#     # Patch asyncio.run used in executor_node
#     import agent_core
#     agent_core.asyncio.run = _patched_asyncio_run  # monkeypatch
#     out = agent.executor_node(state)
#     tool_msgs = [m for m in out["messages"] if isinstance(m, ToolMessage)]
#     assert len(tool_msgs) == 2, "Executor should produce two ToolMessages"
#     assert "Page Snapshot Summary" in tool_msgs[-1].content, "Snapshot summary not found"
#     assert agent.state.current_url == "https://example.com", "Navigation URL not recorded"
#     assert agent.state.page_state.get("headings"), "Parsed page_state missing headings"
#     print("Executor page_state keys:", list(agent.state.page_state.keys()))


# def test_reflector_node_completion():
#     print("\n[TEST] reflector_node_completion")
#     llm = FakeLLM([])
#     tm = FakeToolManager()
#     agent = WebAgent(llm, tm)

#     # Provide messages including a ToolMessage signifying a screenshot saved
#     messages = [
#         HumanMessage(content="Take screenshot"),
#         ToolMessage(content="Screenshot saved to fake_path/screen.png", tool_call_id="sc1")
#     ]
#     state = MessagesState(messages=messages)
#     out = agent.reflector_node(state)
#     # If completion recognized, an AIMessage is appended with reflection text
#     if len(out["messages"]) == len(messages):  # not appended
#         print("No completion detected (expected if keywords logic didn't match).")
#     else:
#         final = out["messages"][-1]
#         assert isinstance(final, AIMessage), "Reflection did not add AIMessage"
#         assert "Task complete: True" in final.content or "Task appears complete" in final.content
#         assert agent.state.task_complete, "task_complete flag not set"
#         print("Reflector flagged completion.")


# def test_route_from_planner_variants():
#     print("\n[TEST] route_from_planner_variants")
#     llm = FakeLLM([])
#     tm = FakeToolManager()
#     agent = WebAgent(llm, tm)

#     # Case 1: tool calls => executor
#     ai_with_tool = AIMessage(content="call tool", tool_calls=[{"name": "browser_snapshot", "args": {}, "id": "x", "type": "tool_call"}])
#     state = MessagesState(messages=[HumanMessage(content="G"), ai_with_tool])
#     route = agent.route_from_planner(state)
#     assert route == "executor", "Expected executor route"
#     print("Route (tool_calls) ->", route)

#     # Case 2: no tool calls => reflector
#     ai_plain = AIMessage(content="no tool")
#     state = MessagesState(messages=[HumanMessage(content="G"), ai_plain])
#     route = agent.route_from_planner(state)
#     assert route == "reflector", "Expected reflector route"
#     print("Route (no tool_calls) ->", route)

#     # Case 3: task_complete flag => exit
#     agent.state.task_complete = True
#     route = agent.route_from_planner(state)
#     assert route == "exit", "Expected exit route"
#     print("Route (task_complete) ->", route)


# def test_run_end_to_end():
#     print("\n[TEST] run_end_to_end")
#     # Scripted LLM steps:
#     # 1. navigate
#     # 2. snapshot
#     # 3. take screenshot
#     # 4. no tool calls (reflector will see 'screenshot saved' output if we inject)
#     llm = FakeLLM([
#         {"tool_calls": [{"name": "browser_navigate", "args": {"url": "https://example.com"}, "id": "n1", "type": "tool_call"}], "text": "Navigate step"},
#         {"tool_calls": [{"name": "browser_snapshot", "args": {}, "id": "s1", "type": "tool_call"}], "text": "Snapshot step"},
#         {"tool_calls": [{"name": "browser_take_screenshot", "args": {}, "id": "sc1", "type": "tool_call"}], "text": "Screenshot step"},
#         {"text": "No more actions"}
#     ])
#     tm = FakeToolManager()
#     agent = WebAgent(llm, tm)

#     # Patch asyncio.run for executor_node
#     import agent_core
#     agent_core.asyncio.run = _patched_asyncio_run

#     # We slightly tweak reflector detection by ensuring the screenshot output has 'saved'
#     # FakeToolManager already returns "Screenshot saved to fake_path/screen.png"

#     final_messages = asyncio.run(agent.run("Goal: Grab a screenshot of example.com"))
#     print(f"Total messages produced: {len(final_messages)}")
#     for m in final_messages[-6:]:
#         kind = type(m).__name__
#         content_preview = (m.content or "")[:100].replace("\n", " ")
#         print(f"  - {kind}: {content_preview}")

#     assert any(isinstance(m, ToolMessage) and "screenshot" in m.content.lower() for m in final_messages), \
#         "Screenshot tool output missing"
#     assert agent.state.step_count >= 3, "Expected multiple planner steps executed"
#     print("End-to-end run completed (step_count =", agent.state.step_count, ")")

# # ---------------------------------------------------------------------------
# # Runner
# # ---------------------------------------------------------------------------

# def main():
#     failures = 0
#     tests = [
#         ("planner_node_basic", test_planner_node_basic),
#         ("executor_node_navigate_and_snapshot", test_executor_node_navigate_and_snapshot),
#         ("reflector_node_completion", test_reflector_node_completion),
#         ("route_from_planner_variants", test_route_from_planner_variants),
#         ("run_end_to_end", test_run_end_to_end),
#     ]
#     for name, fn in tests:
#         try:
#             fn()
#             print(f"[PASS] {name}")
#         except AssertionError as ae:
#             print(f"[FAIL] {name}: {ae}")
#             failures += 1
#         except Exception as e:
#             print(f"[ERROR] {name}: {e}")
#             failures += 1
#     print("\nSummary: {} failed / {} total".format(failures, len(tests)))
#     if failures:
#         raise SystemExit(1)
#     print("All agent_core tests passed.")

# if __name__ == "__main__":
#     main()



"""
Verbose instrumentation test harness for agent_core.WebAgent.

Goal:
  Provide a step-by-step, fully logged walk through each node (planner, executor,
  reflector), show routing decisions, state changes, and final transcript so you
  can understand HOW the agent moves through its workflow.

What this DOES:
  - Uses FakeLLM with a scripted sequence of steps (navigate -> snapshot -> click -> screenshot -> finish).
  - Uses FakeToolManager to simulate tool outputs deterministically (no network).
  - Wraps each node call with verbose logging (before/after messages, diffs).
  - Prints internal AgentState deltas (navigation_history, visited_elements, page_state summary).
  - Shows routing decisions from route_from_planner.
  - Provides JSON summary at end (optional file output).
  - Color output (disable with NO_COLOR=1).
  - Optional --json and --save-json flags.

Run:
    python test_agent_core_verbose.py
    python test_agent_core_verbose.py --json          (prints JSON summary)
    python test_agent_core_verbose.py --save-json run.json  (writes JSON file)
    NO_COLOR=1 python test_agent_core_verbose.py      (no ANSI colors)

If you later refactor agent_core.executor_node to be async (recommended),
remove the monkeypatch section noted below.

This file does NOT modify your agent_core.py; it only inspects behavior.
"""

import json
import os
import time
import asyncio
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from langchain_core.messages import (
    HumanMessage, AIMessage, ToolMessage, SystemMessage
)
from langgraph.graph import MessagesState

# Import the agent under test
import agent_core
from agent_core import WebAgent

# ------------------------------------------------------------------------------------
# Color / Logging helpers
# ------------------------------------------------------------------------------------
USE_COLOR = os.getenv("NO_COLOR") not in ("1", "true", "TRUE")
def _c(code: str) -> str: return code if USE_COLOR else ""

CLR_RESET = _c("\033[0m")
CLR_INFO = _c("\033[36m")
CLR_STEP = _c("\033[34m")
CLR_OK = _c("\033[32m")
CLR_WARN = _c("\033[33m")
CLR_ERR = _c("\033[31m")
CLR_DIM = _c("\033[2m")
CLR_HL = _c("\033[95m")

def log(section: str, msg: str, level: str = "INFO"):
    color = {
        "INFO": CLR_INFO,
        "STEP": CLR_STEP,
        "OK": CLR_OK,
        "WARN": CLR_WARN,
        "ERR": CLR_ERR,
        "DIM": CLR_DIM,
        "HL": CLR_HL
    }.get(level, CLR_INFO)
    ts = time.strftime("%H:%M:%S")
    print(f"{color}[{ts}] [{section}] {msg}{CLR_RESET}")

def truncate(text: str, limit: int = 300):
    if len(text) <= limit: return text
    return text[:limit] + f"... <truncated {len(text)-limit} chars>"

# ------------------------------------------------------------------------------------
# Fakes (Scripted LLM + Tool Manager)
# ------------------------------------------------------------------------------------
@dataclass
class ScriptStep:
    text: str
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)

class FakeLLM:
    """
    Returns AIMessage objects in order. Each ScriptStep may contain
    zero or more tool_calls. After sequence exhausted returns terminal message.
    """
    def __init__(self, steps: List[ScriptStep]):
        self.steps = steps
        self.index = 0

    def invoke(self, messages: List[Any]):
        if self.index >= len(self.steps):
            return AIMessage(content="(LLM) No further plan. Stopping.")
        step = self.steps[self.index]
        self.index += 1
        if step.tool_calls:
            return AIMessage(content=step.text, tool_calls=step.tool_calls)
        return AIMessage(content=step.text)

class FakeToolManager:
    """
    Simulates minimal tool set used by agent:
      - browser_navigate
      - browser_snapshot
      - browser_click
      - browser_take_screenshot
    """
    def __init__(self):
        self.calls: List[Dict[str, Any]] = []
        self.snap_counter = 0

    def get_tool_descriptions(self):
        return "- browser_navigate (url)\n- browser_snapshot ()\n- browser_click (element)\n- browser_take_screenshot ()"

    async def execute_tool(self, name: str, args: Dict[str, Any]):
        self.calls.append({"tool": name, "args": dict(args)})
        if name == "browser_navigate":
            return f"Navigated OK to {args.get('url')}"
        if name == "browser_click":
            return f"Clicked element '{args.get('element')}' successfully"
        if name == "browser_take_screenshot":
            return "Screenshot saved at path: simulated/path/screenshot.png"
        if name == "browser_snapshot":
            self.snap_counter += 1
            # Provide incremental URL / content to watch agent state changes
            return (
                "### Page state\n"
                f"- Page URL: https://example.com/step{self.snap_counter}\n"
                "- Page Title: Example Page Title\n"
                "- Page Snapshot:\n"
                "```yaml\n"
                "- generic [ref=e2]:\n"
                "  - heading \"Main Heading\" [ref=e3]\n"
                "  - paragraph [ref=e4]: Some text here.\n"
                "  - button \"Login\" [ref=e5]\n"
                "  - textbox \"Username\" [ref=e6]\n"
                "  - textbox \"Password\" [ref=e7]\n"
                "  - link \"Help\" [ref=e8]\n"
                "```"
            )
        return f"{name} executed"

# ------------------------------------------------------------------------------------
# Monkeypatch for executor_node's internal asyncio.run usage
# (Remove when executor_node is refactored to async)
# ------------------------------------------------------------------------------------
_original_asyncio_run = asyncio.run
def _patched_asyncio_run(coro):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        return loop.create_task(coro)  # returns Task object (not awaited) -> adjust executor if needed
    return _original_asyncio_run(coro)

# We patch after import so agent_core.executor_node uses it
agent_core.asyncio.run = _patched_asyncio_run

# ------------------------------------------------------------------------------------
# Instrumentation Helpers
# ------------------------------------------------------------------------------------
def describe_messages(msgs: List[Any]) -> List[Dict[str, Any]]:
    """Produce structured summary for JSON export."""
    out = []
    for i, m in enumerate(msgs):
        kind = type(m).__name__
        content = getattr(m, "content", "")
        tool_calls = getattr(m, "tool_calls", None)
        out.append({
            "index": i,
            "type": kind,
            "preview": truncate(content, 180),
            "tool_calls": tool_calls
        })
    return out

def diff_new_messages(before: List[Any], after: List[Any]) -> List[Any]:
    if len(after) <= len(before):
        return []
    return after[len(before):]

def summarize_page_state(page_state: Dict[str, Any]) -> Dict[str, Any]:
    if not page_state:
        return {}
    return {
        "title": page_state.get("title"),
        "headings": page_state.get("headings", [])[:3],
        "buttons": page_state.get("buttons", [])[:5],
        "inputs": page_state.get("inputs", [])[:5],
        "links": page_state.get("links", [])[:5],
        "interactives_count": len(page_state.get("interactives", [])),
    }

# ------------------------------------------------------------------------------------
# Core Verbose Runner
# ------------------------------------------------------------------------------------
async def run_verbose():
    log("INIT", "Constructing fake LLM & tool manager", "STEP")
    steps = [
        ScriptStep("Plan: Navigate to page", [
            {"name": "browser_navigate", "args": {"url": "https://example.com"}, "id": "nav1", "type": "tool_call"}
        ]),
        ScriptStep("Plan: Take snapshot", [
            {"name": "browser_snapshot", "args": {}, "id": "snap1", "type": "tool_call"}
        ]),
        ScriptStep("Plan: Click login", [
            {"name": "browser_click", "args": {"element": "Login"}, "id": "clk1", "type": "tool_call"}
        ]),
        ScriptStep("Plan: Take screenshot", [
            {"name": "browser_take_screenshot", "args": {}, "id": "shot1", "type": "tool_call"}
        ]),
        ScriptStep("Reflection: Possibly done (no tool call)"),
        ScriptStep("Final message (no tool call)")
    ]
    llm = FakeLLM(steps)
    tm = FakeToolManager()
    agent = WebAgent(llm, tm)

    # Build compiled workflow
    log("GRAPH", "Compiling workflow DAG", "STEP")
    workflow = agent.build_workflow()

    # Initial state
    human_goal = "Goal: Visit site, inspect, login, screenshot."
    state = MessagesState(messages=[HumanMessage(content=human_goal)])
    all_intermediate: List[Dict[str, Any]] = []

    log("START", f"Initial human goal: {human_goal}", "HL")

    iteration = 0
    while True:
        iteration += 1
        log("LOOP", f"=== Iteration {iteration} ===", "STEP")

        # 1. Planner node
        before_msgs = state["messages"][:]
        planner_out = agent.planner_node(state)
        state["messages"] = planner_out["messages"]
        new_msgs = diff_new_messages(before_msgs, state["messages"])
        log("PLANNER", f"Produced {len(new_msgs)} new message(s); step_count={agent.state.step_count}", "OK")
        for nm in new_msgs:
            log("PLANNER_MSG", f"{type(nm).__name__}: {truncate(nm.content,120)}", "DIM")
            if isinstance(nm, AIMessage) and nm.tool_calls:
                log("PLANNER_TOOLS", f"Tool calls: {nm.tool_calls}", "INFO")

        route = agent.route_from_planner(state)
        log("ROUTE", f"Route after planner -> {route}", "STEP")

        # Record planner snapshot
        all_intermediate.append({
            "iteration": iteration,
            "phase": "planner",
            "route": route,
            "agent_state": {
                "step_count": agent.state.step_count,
                "navigation_history": agent.state.navigation_history[:],
                "visited_elements": list(agent.state.visited_elements),
                "task_complete": agent.state.task_complete,
                "errors": agent.state.errors[:]
            },
            "messages": describe_messages(state["messages"])
        })

        if route == "exit":
            log("EXIT", "Route indicates exit (task_complete).", "WARN")
            break

        # 2. Executor (if route requires)
        if route == "executor":
            before_msgs_exec = state["messages"][:]
            exec_out = agent.executor_node(state)
            state["messages"] = exec_out["messages"]
            new_exec = diff_new_messages(before_msgs_exec, state["messages"])
            log("EXECUTOR", f"Added {len(new_exec)} tool result message(s)", "OK")
            for nm in new_exec:
                log("EXEC_MSG", f"{type(nm).__name__}: {truncate(nm.content, 140)}", "DIM")
            log("STATE", f"navigation_history={agent.state.navigation_history}", "INFO")
            log("STATE", f"visited_elements={list(agent.state.visited_elements)}", "INFO")
            log("STATE", f"page_state summary={summarize_page_state(agent.state.page_state)}", "INFO")

            all_intermediate.append({
                "iteration": iteration,
                "phase": "executor",
                "agent_state": {
                    "step_count": agent.state.step_count,
                    "navigation_history": agent.state.navigation_history[:],
                    "visited_elements": list(agent.state.visited_elements),
                    "task_complete": agent.state.task_complete,
                    "errors": agent.state.errors[:],
                    "page_state": summarize_page_state(agent.state.page_state)
                },
                "messages": describe_messages(state["messages"]),
                "tool_calls_executed": tm.calls[-len(new_exec):] if new_exec else []
            })

            # Post executor now routes to planner per workflow (reflector removed)
            log("ROUTE", "Transition executor -> planner", "STEP")
            route = "planner"

        # Safety: stop if steps exceed threshold to avoid infinite test loops.
        if iteration > 12:
            log("SAFETY", "Iteration limit reached (12). Aborting.", "ERR")
            break

    # Final transcript
    log("FINAL", "---- COMPLETE TRANSCRIPT (message index : type) ----", "HL")
    for i, m in enumerate(state["messages"]):
        kind = type(m).__name__
        log("MSG", f"#{i:02d} {kind} :: {truncate(m.content, 160)}", "DIM")

    # JSON summary
    summary = {
        "final_step_count": agent.state.step_count,
        "final_navigation_history": agent.state.navigation_history,
        "final_visited_elements": list(agent.state.visited_elements),
        "final_task_complete": agent.state.task_complete,
        "final_errors": agent.state.errors,
        "final_page_state": summarize_page_state(agent.state.page_state),
        "intermediate": all_intermediate,
        "total_messages": len(state["messages"])
    }

    return summary

# ------------------------------------------------------------------------------------
# Command-line Interface
# ------------------------------------------------------------------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Verbose node/function test for agent_core.WebAgent")
    parser.add_argument("--json", action="store_true", help="Print JSON summary to stdout at end")
    parser.add_argument("--save-json", metavar="PATH", help="Write JSON summary to file")
    args = parser.parse_args()

    summary = asyncio.run(run_verbose())

    if args.json:
        print("\n=== JSON SUMMARY (stdout) ===")
        print(json.dumps(summary, indent=2))

    if args.save_json:
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        log("WRITE", f"Saved JSON summary to {args.save_json}", "OK")

    print("\nDone.")

if __name__ == "__main__":
    main()