"""Dynamic tool discovery and execution for the agent."""
""" 
## Function Usage in the Architecture

Here's how each function in the `ToolManager` class is used in the broader architecture:

1. **`__init__(session)`**
   - **Where used:** When the agent is initialized in `main.py`
   - **Purpose:** Creates a new ToolManager instance with the MCP session

2. **`initialize_tools()`**
   - **Where used:** In `run_automation()` function in `main.py` during startup
   - **Purpose:** Discovers all available Playwright MCP tools and creates LangChain wrappers
   - **How it works:** Calls `session.list_tools()` to get all available tools, then creates wrappers for each tool using `_create_tool_wrapper()`

3. **`_create_tool_wrapper(name, description, schema)`**
   - **Where used:** Internally by `initialize_tools()`
   - **Purpose:** Creates a LangChain tool wrapper for an MCP tool
   - **How it works:** 
     - Builds appropriate type annotations based on parameter schema
     - Creates a dynamic async function that calls the MCP tool
     - Wraps it as a LangChain tool with proper metadata

4. **`get_tool_descriptions()`**
   - **Where used:** In `planner_node()` method of `WebAgent` class in `agent_core.py`
   - **Purpose:** Generates formatted descriptions of all tools for the system prompt
   - **How it works:** Formats each tool's name, description, and parameters into a readable string

5. **`execute_tool(name, args)`**
   - **Where used:** In `executor_node()` method of `WebAgent` class in `agent_core.py`
   - **Purpose:** Executes a specific MCP tool with the given arguments
   - **How it works:**
     - Handles parameter synonyms (e.g., "selector" → "element")
     - Calls the tool via the MCP session
     - Processes and formats the result
     - Handles errors gracefully


This `ToolManager` class is a crucial component that bridges between:
1. The MCP Playwright tools that perform actual browser automation
2. The LangChain tools framework that the LLM agent can use


The dynamic tool discovery approach means the agent can automatically adapt to any new tools that might be added to the MCP system without code changes, making it highly extensible.
"""


import re
import os
import time
import json
from typing import List, Dict, Any
from dataclasses import dataclass, field

from langchain_core.tools import BaseTool, tool
from mcp import ClientSession


class ToolManager:
    """Manages dynamic tool discovery and execution for the web agent."""
    
    def __init__(self, session: ClientSession):
        self.session = session
        self.tools: List[BaseTool] = []
        self.tool_schemas: Dict[str, Dict[str, Any]] = {}
        
    async def initialize_tools(self):
        """Discover and initialize all available tools from the MCP session."""
        listing = await self.session.list_tools()
        
        for tool_info in getattr(listing, "tools", []) or []:
            name = getattr(tool_info, "name", None)
            if not name:
                continue
                
            description = getattr(tool_info, "description", "") or f"MCP tool: {name}"
            schema = getattr(tool_info, "input_schema", None) or getattr(tool_info, "inputSchema", None) or {}
            
            # Create wrapper for this tool
            wrapper = self._create_tool_wrapper(name, description, schema)
            self.tools.append(wrapper)
            
            # Store schema for validation
            props = schema.get("properties", {}) if isinstance(schema, dict) else {}
            required = set(schema.get("required", []) if isinstance(schema, dict) else [])
            
            self.tool_schemas[name] = {
                "properties": props,
                "required": required,
                "description": description
            }
        return self.tools
        
    def _create_tool_wrapper(self, name: str, description: str, schema: Dict[str, Any]) -> BaseTool:
        """Create a langchain tool wrapper for an MCP tool."""
        props = schema.get("properties", {}) if isinstance(schema, dict) else {}
        
        # Create type annotations
        annotations: Dict[str, Any] = {}
        for prop_name, prop_def in props.items():
            prop_type = prop_def.get("type") if isinstance(prop_def, dict) else None
            annotations[prop_name] = {
                "integer": int,
                "number": float, 
                "boolean": bool,
                "array": list
            }.get(prop_type, str)
        
        async def tool_function(**kwargs):
            """Dynamic tool wrapper function."""
            # Unwrap nested call_kwargs if present
            if "call_kwargs" in kwargs and isinstance(kwargs["call_kwargs"], dict) and len(kwargs) == 1:
                kwargs = kwargs["call_kwargs"]
            
            # Handle kwargs string
            if "kwargs" in kwargs and isinstance(kwargs["kwargs"], str):
                try:
                    kwargs = json.loads(kwargs["kwargs"])
                except:
                    pass
                    
            # Call the actual MCP tool
            result = await self.session.call_tool(name, kwargs)
            
            # Process result
            content = getattr(result, "content", None)
            if isinstance(content, list):
                collected = []
                for item in content:
                    text = getattr(item, "text", None)
                    if text:
                        collected.append(text)
                return "\n".join(collected) if collected else str(result)
            return str(result)
            
        # Create proper tool name for Python
        py_name = re.sub(r"[^0-9a-zA-Z_]+", "_", name)
        
        # Set function metadata
        tool_function.__name__ = py_name
        tool_function.__doc__ = description
        tool_function.__annotations__ = annotations  # type: ignore
        
        # Wrap as langchain tool
        wrapped = tool(tool_function)
        wrapped.name = name  # type: ignore
        wrapped.description = description  # type: ignore
        
        return wrapped
        
    def get_tool_descriptions(self) -> str:
        """Get formatted descriptions of all available tools."""
        lines = ["Available tools:"]
        
        for name, schema in self.tool_schemas.items():
            desc = schema["description"]
            props = schema["properties"]
            required = schema["required"]
            
            # Format parameters
            param_parts = []
            for param_name, param_def in props.items():
                param_type = param_def.get("type", "any") if isinstance(param_def, dict) else "any"
                # Make only 'ref' required for browser_click
                req_mark = "*" if (param_name in required and not (name == "browser_click" and param_name == "element")) else ""
                param_parts.append(f"{param_name}:{param_type}{req_mark}")
                
            param_str = ", ".join(param_parts) if param_parts else "No parameters"
            lines.append(f"- {name}: {desc[:100]} | Parameters: {param_str}")
            
        return "\n".join(lines)
        
        

    async def execute_tool(self, name: str, args: Dict[str, Any]) -> str:
        """Execute a tool with the given arguments."""

        # Special handling for browser_click to make it easier
        if name == "browser_click":
            # If we only have text but no ref, try to find the ref first
            if ("element" in args or "selector" in args) and "ref" not in args:
                # Get the element text
                element_text = args.get("element") or args.get("selector", "")
                
                # First try to get a snapshot if we don't have one
                snapshot_result = await self.session.call_tool("browser_snapshot", {})
                snapshot_text = str(getattr(snapshot_result, "content", None) or "")
                
                # Try to find the reference
                from utils import find_element_ref
                ref = find_element_ref(snapshot_text, element_text)
                if ref:
                    # Found a reference, use it instead
                    args = {"ref": ref}
                    print(f"Found reference {ref} for element '{element_text}'")

        # Handle file upload parameter conversion
        if name == "browser_file_upload":
            print(f"[DEBUG] Processing file upload with args: {args}")
            
            # Get and validate file path
            file_path = args.get("filePath") or args.get("path")
            if not file_path:
                return "Error: No file path provided for upload"
                
            # Ensure file exists
            if not os.path.exists(file_path):
                return f"Error: File not found at path: {file_path}"
            
            # Prepare upload args with absolute path
            file_path = os.path.abspath(file_path)
            upload_args = {"paths": [file_path]}
            
            # Add ref if provided
            if "ref" in args and args["ref"]:
                ref = args["ref"]
                upload_args["ref"] = ref
                
                print(f"[DEBUG] Attempting to click file input with ref: {ref}")
                
                try:
                    # Try to find the element text from page state
                    element_text = f"ref:{ref}"  # Default element text
                    if hasattr(self, 'state') and hasattr(self.state, 'page_state') and 'refs' in self.state.page_state:
                        for text, refs in self.state.page_state['refs'].items():
                            if ref in refs:
                                element_text = text
                                print(f"[DEBUG] Found element text for ref {ref}: {element_text}")
                                break
                    
                    # Click the file input element
                    click_result = await self.session.call_tool(
                        "browser_click", 
                        {"ref": ref, "element": element_text}
                    )
                    print(f"[DEBUG] Click result: {click_result}")
                    
                    # Small delay to ensure the file dialog is ready
                    await asyncio.sleep(1)
                    
                except Exception as e:
                    print(f"[WARNING] Click before upload failed, continuing with upload: {str(e)}")
            
            # Update args for the actual upload
            print(f"[DEBUG] Executing file upload with args: {upload_args}")
            args = upload_args
        
        # Adapt common parameter synonyms
        if name in self.tool_schemas:
            schema = self.tool_schemas[name]
            props = schema.get("properties", {})
            
            # Handle selector -> element synonym
            if "element" in props and "element" not in args and "selector" in args:
                args["element"] = args.pop("selector")
                
        # Execute the tool
        try:
            result = await self.session.call_tool(name, args)
            print("?????????????? \n" , "Tool Call:", name , "\n Result = " , result)
            
            # Process result
            content = getattr(result, "content", None)
            if isinstance(content, list):
                collected = []
                for item in content:
                    text = getattr(item, "text", None)
                    if text:
                        collected.append(text)
                return "\n".join(collected) if collected else str(result)
            return str(result)
        except Exception as e:
            return f"Error executing {name}: {str(e)}"
        





# ===================================================
# Enhanced Embedded Tests (Verbose)
# ===================================================
import asyncio
from mcp.client.stdio import stdio_client
from mcp import StdioServerParameters

# Colors (optional)
USE_COLOR = os.getenv("NO_COLOR") not in ("1", "true", "TRUE")

def _c(code: str) -> str:
    if not USE_COLOR:
        return ""
    return code

COLOR_RESET = _c("\033[0m")
COLOR_BLUE = _c("\033[34m")
COLOR_GREEN = _c("\033[32m")
COLOR_RED = _c("\033[31m")
COLOR_YELLOW = _c("\033[33m")
COLOR_CYAN = _c("\033[36m")
COLOR_DIM = _c("\033[2m")


def log(section: str, message: str, level: str = "INFO"):
    ts = time.strftime("%H:%M:%S")
    color = {
        "INFO": COLOR_BLUE,
        "OK": COLOR_GREEN,
        "WARN": COLOR_YELLOW,
        "ERROR": COLOR_RED,
        "STEP": COLOR_CYAN,
        "DATA": COLOR_DIM,
    }.get(level, COLOR_BLUE)
    print(f"{color}[{ts}] [{section}] {message}{COLOR_RESET}")


def truncate(text: str, limit: int = 400) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"... <truncated {len(text)-limit} chars>"


def extract_counts(snapshot: str) -> Dict[str, int]:
    lower = snapshot.lower()
    counts = {}
    counts["buttons"] = lower.count(" button ")
    counts["inputs"] = sum(lower.count(k) for k in [" textbox ", " input ", " combobox ", " textarea "])
    counts["links"] = lower.count(" link ")
    counts["tabs"] = lower.count(" tab ")
    return counts


def extract_screenshot_path(output: str) -> str | None:
    # Common patterns: saved to path/to/file.png OR screenshot: path
    path_match = re.search(r'((?:[A-Z]:)?[\\/\w\-.]+\.png)', output, re.IGNORECASE)
    if path_match:
        return path_match.group(1)
    path_match = re.search(r'((?:[A-Z]:)?[\\/\w\-.]+\.jpe?g)', output, re.IGNORECASE)
    if path_match:
        return path_match.group(1)
    return None


@dataclass
class StepLog:
    tool: str
    args: Dict[str, Any]
    ms: float
    output_truncated: str
    screenshot_path: str | None = None
    snapshot_counts: Dict[str, int] = field(default_factory=dict)


@dataclass
class TestResult:
    name: str
    passed: bool
    detail: str = ""
    steps: List[StepLog] = field(default_factory=list)


SERVER_PARAMS = StdioServerParameters(
    command = "/Users/arhan/.nvm/versions/node/v22.18.0/bin/npx",
    args=["-y", "@playwright/mcp@latest"],
)


async def execute_and_log(tm: ToolManager, tool: str, args: Dict[str, Any], capture_snapshot_counts=False) -> StepLog:
    start = time.perf_counter()
    result = await tm.execute_tool(tool, args)
    elapsed = (time.perf_counter() - start) * 1000
    trunc = truncate(result)
    screenshot_path = extract_screenshot_path(result) if tool == "browser_take_screenshot" else None
    snapshot_counts = extract_counts(result) if capture_snapshot_counts else {}
    log("TOOL", f"{tool} args={args} took {elapsed:.1f} ms", "STEP")
    log("OUTPUT", trunc, "DATA")
    if screenshot_path:
        log("SCREENSHOT", f"Detected file: {screenshot_path}", "OK")
    if capture_snapshot_counts:
        log("SNAPSHOT", f"Counts: {snapshot_counts}", "INFO")
    return StepLog(tool=tool, args=args, ms=elapsed, output_truncated=trunc,
                   screenshot_path=screenshot_path, snapshot_counts=snapshot_counts)


async def test_initialize_tools() -> TestResult:
    steps: List[StepLog] = []
    try:
        log("TEST", "initialize_tools: starting")
        log("CONNECT", "🔌 Connecting to Playwright MCP server...")
        async with stdio_client(SERVER_PARAMS) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tm = ToolManager(session)
                start = time.perf_counter()
                tools = await tm.initialize_tools()
                elapsed = (time.perf_counter() - start) * 1000
                detail = f"Discovered {len(tools)} tools in {elapsed:.1f} ms"
                log("DISCOVERY", detail, "OK")
                desc = tm.get_tool_descriptions()
                assert tools, "No tools discovered"
                assert "Available tools:" in desc
                return TestResult("initialize_tools", True, detail, steps)
    except AssertionError as ae:
        log("ERROR", str(ae), "ERROR")
        return TestResult("initialize_tools", False, f"Assertion failed: {ae}", steps)
    except Exception as e:
        log("ERROR", repr(e), "ERROR")
        return TestResult("initialize_tools", False, f"Exception: {e}", steps)


async def test_navigation_and_snapshot(url: str) -> TestResult:
    steps: List[StepLog] = []
    name = f"navigation_and_snapshot:{url}"
    try:
        log("TEST", f"{name} starting")
        log("CONNECT", "🔌 Connecting to Playwright MCP server...")
        async with stdio_client(SERVER_PARAMS) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tm = ToolManager(session)
                await tm.initialize_tools()

                if "browser_navigate" not in tm.tool_schemas:
                    return TestResult(name, False, "browser_navigate tool missing", steps)
                if "browser_snapshot" not in tm.tool_schemas:
                    return TestResult(name, False, "browser_snapshot tool missing", steps)

                steps.append(await execute_and_log(tm, "browser_navigate", {"url": url}))
                snap_step = await execute_and_log(tm, "browser_snapshot", {}, capture_snapshot_counts=True)
                steps.append(snap_step)

                snap_res_lower = snap_step.output_truncated.lower()
                assert len(snap_step.output_truncated) > 0, "Empty snapshot"
                assert any(t in snap_res_lower for t in ["title", "button", "input", "link", "page"]), \
                    "Snapshot missing structural hints"

                detail = f"Snapshot OK (buttons={snap_step.snapshot_counts.get('buttons',0)}, " \
                         f"inputs={snap_step.snapshot_counts.get('inputs',0)})"
                return TestResult(name, True, detail, steps)
    except AssertionError as ae:
        log("ERROR", str(ae), "ERROR")
        return TestResult(name, False, f"{ae}", steps)
    except Exception as e:
        log("ERROR", repr(e), "ERROR")
        return TestResult(name, False, f"Exception: {e}", steps)


async def test_click_with_selector(url: str) -> TestResult:
    steps: List[StepLog] = []
    name = "click_with_selector"
    try:
        log("TEST", f"{name} starting on {url}")
        log("CONNECT", "🔌 Connecting to Playwright MCP server...")
        async with stdio_client(SERVER_PARAMS) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tm = ToolManager(session)
                await tm.initialize_tools()

                if "browser_click" not in tm.tool_schemas:
                    return TestResult(name, True, "browser_click not available (skipped)", steps)

                steps.append(await execute_and_log(tm, "browser_navigate", {"url": url}))
                snap_step = await execute_and_log(tm, "browser_snapshot", {}, capture_snapshot_counts=True)
                steps.append(snap_step)

                labels = re.findall(r'"([^"]{1,60})"', snap_step.output_truncated)
                candidates = [l for l in labels if l and len(l.split()) <= 5]
                if not candidates:
                    return TestResult(name, True, "No clickable labels (skipped)", steps)

                target = candidates[0]
                click_step = await execute_and_log(tm, "browser_click", {"selector": target})
                steps.append(click_step)

                if "Error executing" in click_step.output_truncated or "CLICK_ERROR" in click_step.output_truncated:
                    return TestResult(name, False, f"Click error: {click_step.output_truncated[:200]}", steps)
                return TestResult(name, True, f"Clicked/attempted '{target}'", steps)
    except Exception as e:
        log("ERROR", repr(e), "ERROR")
        return TestResult(name, False, f"Exception: {e}", steps)


async def test_screenshot(url: str) -> TestResult:
    steps: List[StepLog] = []
    name = "screenshot"
    try:
        log("TEST", f"{name} starting on {url}")
        log("CONNECT", "🔌 Connecting to Playwright MCP server...")
        async with stdio_client(SERVER_PARAMS) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tm = ToolManager(session)
                await tm.initialize_tools()

                if "browser_take_screenshot" not in tm.tool_schemas:
                    return TestResult(name, True, "browser_take_screenshot not available", steps)

                steps.append(await execute_and_log(tm, "browser_navigate", {"url": url}))
                # Optional snapshot before screenshot
                if "browser_snapshot" in tm.tool_schemas:
                    steps.append(await execute_and_log(tm, "browser_snapshot", {}, capture_snapshot_counts=True))

                shot_step = await execute_and_log(tm, "browser_take_screenshot", {})
                steps.append(shot_step)

                if "Error executing" in shot_step.output_truncated:
                    return TestResult(name, False, f"Screenshot error: {shot_step.output_truncated[:200]}", steps)

                path = shot_step.screenshot_path or "path-not-detected"
                return TestResult(name, True, f"Screenshot captured (path={path})", steps)
    except Exception as e:
        log("ERROR", repr(e), "ERROR")
        return TestResult(name, False, f"Exception: {e}", steps)


async def run_all_tests():
    test_sites = [
        "https://example.com",
        "https://httpbin.org/forms/post",
        "https://www.wikipedia.org",
    ]
    results: List[TestResult] = []

    results.append(await test_initialize_tools())

    for site in test_sites:
        results.append(await test_navigation_and_snapshot(site))

    results.append(await test_click_with_selector(test_sites[0]))
    results.append(await test_screenshot(test_sites[1]))

    # Summary
    print("\n" + "=" * 68)
    print("TEST SUMMARY")
    print("=" * 68)
    passed = 0
    summary_payload = []
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        color = COLOR_GREEN if r.passed else COLOR_RED
        print(f"{color}[{status}]{COLOR_RESET} {r.name}: {r.detail}")
        if r.passed:
            passed += 1
        summary_payload.append({
            "name": r.name,
            "passed": r.passed,
            "detail": r.detail,
            "steps": [
                {
                    "tool": s.tool,
                    "args": s.args,
                    "ms": round(s.ms, 2),
                    "screenshot_path": s.screenshot_path,
                    "snapshot_counts": s.snapshot_counts,
                } for s in r.steps
            ]
        })
    total = len(results)
    print(f"\nResult: {passed}/{total} tests passed")

    if os.getenv("TEST_JSON_SUMMARY") in ("1", "true", "TRUE"):
        json_path = "test_results_summary.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary_payload, f, indent=2)
        print(f"\nJSON summary written to {json_path}")

    return 0 if passed == total else 1


def _run(coro):
    return asyncio.run(coro)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run verbose ToolManager tests")
    parser.add_argument("--run-tests", action="store_true", help="Execute all embedded tests")
    parser.add_argument(
        "--single-test",
        choices=["init", "nav", "click", "screenshot"],
        help="Run only a specific test (uses provided or default URL)",
    )
    parser.add_argument("--url", help="Override URL for nav/click/screenshot single test")
    parser.add_argument("--json", action="store_true", help="Emit JSON summary (same as env TEST_JSON_SUMMARY=1)")
    args = parser.parse_args()

    if args.json:
        os.environ["TEST_JSON_SUMMARY"] = "1"

    if args.run_tests:
        exit_code = _run(run_all_tests())
        raise SystemExit(exit_code)

    if args.single_test:
        site_default = "https://example.com"
        url = args.url or site_default
        if args.single_test == "init":
            res = _run(test_initialize_tools())
            raise SystemExit(0 if res.passed else 1)
        elif args.single_test == "nav":
            res = _run(test_navigation_and_snapshot(url))
            raise SystemExit(0 if res.passed else 1)
        elif args.single_test == "click":
            res = _run(test_click_with_selector(url))
            raise SystemExit(0 if res.passed else 1)
        elif args.single_test == "screenshot":
            res = _run(test_screenshot(url))
            raise SystemExit(0 if res.passed else 1)

    print(
        "Usage:\n"
        "  python tool_manager.py --run-tests\n"
        "  python tool_manager.py --single-test nav --url https://example.com\n"
        "Options:\n"
        "  --json  Emit JSON summary file\n"
        "Environment:\n"
        "  NO_COLOR=1 disables ANSI colors\n"
        "  TEST_JSON_SUMMARY=1 writes test_results_summary.json\n"

    )
