"""Simplified tool manager for MCP Playwright integration."""
import re
import os
import json
from typing import List, Dict, Any

from langchain_core.tools import BaseTool, tool
from mcp import ClientSession

class ToolManager:
    """Simplified tool manager for web automation."""
    
    def __init__(self, session: ClientSession):
        self.session = session
        self.tools: List[BaseTool] = []
        self.tool_schemas: Dict[str, Dict[str, Any]] = {}
        
    async def initialize_tools(self):
        """Initialize tools from MCP session."""
        listing = await self.session.list_tools()
        
        for tool_info in getattr(listing, "tools", []) or []:
            name = getattr(tool_info, "name", None)
            if not name:
                continue
                
            description = getattr(tool_info, "description", "") or f"Tool: {name}"
            schema = getattr(tool_info, "input_schema", None) or {}
            
            wrapper = self._create_tool_wrapper(name, description, schema)
            self.tools.append(wrapper)
            
            props = schema.get("properties", {}) if isinstance(schema, dict) else {}
            required = set(schema.get("required", []) if isinstance(schema, dict) else [])
            
            self.tool_schemas[name] = {
                "properties": props,
                "required": required,
                "description": description
            }
        return self.tools
        
    def _create_tool_wrapper(self, name: str, description: str, schema: Dict[str, Any]) -> BaseTool:
        """Create a tool wrapper."""
        props = schema.get("properties", {}) if isinstance(schema, dict) else {}
        
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
            """Execute MCP tool."""
            if "call_kwargs" in kwargs and len(kwargs) == 1:
                kwargs = kwargs["call_kwargs"]
            
            if "kwargs" in kwargs and isinstance(kwargs["kwargs"], str):
                try:
                    kwargs = json.loads(kwargs["kwargs"])
                except:
                    pass
                    
            result = await self.session.call_tool(name, kwargs)
            
            content = getattr(result, "content", None)
            if isinstance(content, list):
                collected = []
                for item in content:
                    text = getattr(item, "text", None)
                    if text:
                        collected.append(text)
                return "\n".join(collected) if collected else str(result)
            return str(result)
            
        py_name = re.sub(r"[^0-9a-zA-Z_]+", "_", name)
        tool_function.__name__ = py_name
        tool_function.__doc__ = description
        tool_function.__annotations__ = annotations
        
        wrapped = tool(tool_function)
        wrapped.name = name
        wrapped.description = description
        
        return wrapped
        
    async def execute_tool(self, name: str, args: Dict[str, Any]) -> str:
        """Execute a tool with given arguments."""
        # Handle browser_click simplification
        if name == "browser_click":
            if ("element" in args or "selector" in args) and "ref" not in args:
                element_text = args.get("element") or args.get("selector", "")
                snapshot_result = await self.session.call_tool("browser_snapshot", {})
                snapshot_text = str(getattr(snapshot_result, "content", None) or "")
                
                from utils import find_element_ref
                ref = find_element_ref(snapshot_text, element_text)
                if ref:
                    args = {"ref": ref}
                    print(f"Found ref {ref} for element '{element_text}'")

        # Handle file upload
        if name == "browser_file_upload":
            file_path = args.get("filePath") or args.get("path") or args.get("paths", [None])[0]
            if not file_path:
                return "Error: No file path provided"
                
            if not os.path.exists(file_path):
                return f"Error: File not found at {file_path}"
            
            file_path = os.path.abspath(file_path)
            upload_args = {"paths": [file_path]}
            
            if "ref" in args and args["ref"]:
                upload_args["ref"] = args["ref"]
                try:
                    element_text = f"ref:{args['ref']}"
                    click_result = await self.session.call_tool(
                        "browser_click", 
                        {"ref": args["ref"], "element": element_text}
                    )
                except Exception as e:
                    print(f"Click before upload failed: {e}")
            
            args = upload_args
        
        # Handle parameter synonyms
        if name in self.tool_schemas:
            schema = self.tool_schemas[name]
            props = schema.get("properties", {})
            
            if "element" in props and "element" not in args and "selector" in args:
                args["element"] = args.pop("selector")
                
        try:
            result = await self.session.call_tool(name, args)
            
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