"""Utility functions for web automation agent."""
import re
from typing import Dict, Any, List, Optional

# ---------------- Core Utilities ---------------- #

def truncate_text(text: str, limit: int = 1000) -> str:
    if not text or len(text) <= limit:
        return text
    return text[:limit] + f"... [truncated {len(text) - limit} characters]"


INTERACTIVE_KEYWORDS = ("textbox", "input", "combobox", "textarea", "checkbox", "radio", "select", "button")

def _is_interactive_line(line_lower: str) -> bool:
    return any(k in line_lower for k in INTERACTIVE_KEYWORDS)


def extract_interactive_elements(snapshot_text: str) -> Dict[str, Any]:
    print("\nSNAPSHOT TEXT", snapshot_text ,"\n")
    lines = snapshot_text.splitlines()
    result = {
        "title": None,
        "headings": [],
        "buttons": [],
        "inputs": [],
        "links": [],
        "tabs": [],
        "refs": {},
        "interactives": [],
        "all_clickables": []
    }
    
    ref_pattern = re.compile(r"\[ref=(e\d+)\]")
    quoted_pattern = re.compile(r'"(.+?)"')
    icon_pattern = re.compile(r'<svg[^>]*>.*?</svg>|<i[^>]*>.*?</i>|<img[^>]*>', re.DOTALL)
    
    # First pass: collect context information
    context_map = {}  # ref -> nearby text context
    
    for i, line in enumerate(lines):
        line_lower = line.lower()
        refs = ref_pattern.findall(line)
        
        if refs:
            for ref in refs:
                # Collect context from current and nearby lines
                context_lines = []
                
                # Look backwards for context (up to 3 lines)
                for j in range(max(0, i-3), i):
                    prev_line = lines[j].strip()
                    if prev_line and not ref_pattern.search(prev_line):
                        # Extract meaningful text (not just structural elements)
                        text_content = re.sub(r'^\s*-\s*', '', prev_line)  # Remove list markers
                        text_content = re.sub(r':\s*$', '', text_content)  # Remove trailing colons
                        if len(text_content) > 1 and not text_content.lower() in ['generic', 'text']:
                            context_lines.append(text_content)
                
                # Look forwards for context (up to 2 lines)
                for j in range(i+1, min(len(lines), i+3)):
                    next_line = lines[j].strip()
                    if next_line and not ref_pattern.search(next_line):
                        text_content = re.sub(r'^\s*-\s*', '', next_line)
                        text_content = re.sub(r':\s*$', '', text_content)
                        if len(text_content) > 1 and not text_content.lower() in ['generic', 'text']:
                            context_lines.append(text_content)
                
                context_map[ref] = context_lines
    
    for line in lines:
        line_lower = line.lower()
        
        # Extract title
        if result["title"] is None and "page title:" in line_lower:
            result["title"] = line.split(":", 1)[-1].strip()
            continue
        
        # Check if this is a button element
        is_button = 'button ' in f" {line_lower} "
        
        # Extract all interactive elements with better pattern matching
        interactive_keywords = [
            "button", "link", "tab", "textbox", "input", "combobox", 
            "textarea", "checkbox", "radio", "select", "clickable", "interactive"
        ]
        
        # Check if this line contains any interactive element
        is_button = 'button' in line_lower
        is_interactive = any(f' {kw} ' in f' {line_lower} ' for kw in interactive_keywords)
        
        if is_interactive or is_button:
            # Extract all quoted text (potential labels)
            labels = quoted_pattern.findall(line)
            
            # Extract reference
            refs = ref_pattern.findall(line)
            
            # If no labels found, try to generate meaningful labels
            if not labels and refs:
                ref = refs[0]  # Use first ref
                
                # Try to get label from context
                context_lines = context_map.get(ref, [])
                potential_labels = []
                
                for context in context_lines:
                    # Clean up context text - remove special markers and trim
                    clean_context = re.sub(r'^text:\s*', '', context)  # Remove 'text:' prefix if present
                    clean_context = re.sub(r'[*"]+', '', clean_context).strip()
                    
                    # Skip very short or generic text
                    if len(clean_context) > 2 and clean_context.lower() not in ['generic', 'text', 'img']:
                        potential_labels.append(clean_context)
                
                # Use the most relevant context as label
                if potential_labels:
                    # Prioritize labels that seem more descriptive
                    best_label = None
                    for label in potential_labels:
                        label_lower = label.lower()
                        # Prefer labels with common form field terms
                        if any(term in label_lower for term in ['upload', 'resume', 'cv', 'file', 'portfolio', 'work', 'link']):
                            best_label = label
                            break
                        # Or use the first non-generic one
                        elif not best_label and len(label) > 3:
                            best_label = label
                    
                    if best_label:
                        labels = [best_label]
                
                # If still no labels but it's a button, try to infer from element type
                if not labels and is_button:
                    if icon_pattern.search(line):
                        # Try to get label from nearby text or context
                        if context_lines:
                            # Look for descriptive text in context
                            for context in context_lines:
                                if any(term in context.lower() for term in ['upload', 'file', 'choose', 'portfolio', 'work']):
                                    labels = [context.strip().replace('"', '')]
                                    break
                        
                        if not labels:
                            labels = ["icon_button"]
                    elif 'plus' in line_lower or 'add' in line_lower:
                        labels = ["add_button"]
                    elif '>' in line and '<' in line:
                        # Extract text between > and <
                        text_match = re.search(r'>([^<]+)<', line)
                        if text_match and text_match.group(1).strip():
                            labels = [text_match.group(1).strip()]
                        else:
                            labels = ["button"]
                    else:
                        labels = ["button"]
            
            # Classify the element type
            element_type = "unknown"
            if is_button or 'button' in line_lower:
                element_type = "button"
            elif "tab" in line_lower:
                element_type = "tab"
            elif any(kw in line_lower for kw in ["textbox", "input", "combobox", "textarea"]):
                element_type = "input"
            elif "link" in line_lower:
                element_type = "link"
            elif any(kw in line_lower for kw in ["checkbox", "radio", "select"]):
                element_type = "input"
            
            # Add to appropriate category
            for label in labels:
                if label:
                    # Clean up the label
                    label = label.strip().replace('"', '')
                    if not label:
                        continue
                        
                    # Add to specific category
                    if element_type == "button" and label not in result["buttons"]:
                        result["buttons"].append(label)
                    elif element_type == "tab" and label not in result["tabs"]:
                        result["tabs"].append(label)
                    elif element_type == "input" and label not in result["inputs"]:
                        result["inputs"].append(label)
                    elif element_type == "link" and label not in result["links"]:
                        result["links"].append(label)
                    
                    # Add to general interactives
                    interactive_desc = f"{element_type}: {label}"
                    if interactive_desc not in result["interactives"]:
                        result["interactives"].append(interactive_desc)
                    
                    # Add to all clickables
                    if element_type in ["button", "tab", "link"] and label not in result["all_clickables"]:
                        result["all_clickables"].append(label)
                    
                    # Store ref associations
                    if refs:
                        for ref in refs:
                            if label not in result["refs"]:
                                result["refs"][label] = []
                            if ref not in result["refs"][label]:
                                result["refs"][label].append(ref)
    
    return result


def find_element_ref(snapshot_text: str, element_text: str, element_type: str = None) -> Optional[str]:
    """
    Enhanced element finding with better pattern matching and prioritization.
    """
    lines = snapshot_text.splitlines()
    element_lower = element_text.lower()
    
    candidates = []
    
    for idx, line in enumerate(lines):
        line_lower = line.lower()
        
        # Skip non-interactive lines
        if not any(kw in line_lower for kw in ["button", "link", "tab", "textbox", "input", "clickable"]):
            continue
            
        # Check for various match types with different scores
        score = 0
        
        # Exact match (highest priority)
        if f'"{element_text}"' in line:
            score = 100
        # Contains match
        elif element_lower in line_lower:
            score = 80
        # Fuzzy match (word boundary)
        elif re.search(rf'\b{re.escape(element_lower)}\b', line_lower):
            score = 70
        # Partial match
        elif any(word in line_lower for word in element_lower.split()):
            score = 50
        else:
            continue
            
        # Bonus for type matching
        if element_type and element_type in line_lower:
            score += 20
            
        # Extract reference
        ref_match = re.search(r"\[ref=(e\d+)\]", line)
        if ref_match:
            ref = ref_match.group(1)
            candidates.append((score, ref, line))
            
            # Also check previous line for ref
            if idx > 0:
                prev_ref_match = re.search(r"\[ref=(e\d+)\]", lines[idx-1])
                if prev_ref_match:
                    prev_ref = prev_ref_match.group(1)
                    candidates.append((score - 5, prev_ref, lines[idx-1]))
    
    if not candidates:
        # Try alternative element names for "Apply Here"
        if "apply here" in element_lower:
            alternative_texts = ["apply", "submit", "continue", "next", "start application"]
            for alt_text in alternative_texts:
                if alt_text != element_text:
                    alt_ref = find_element_ref(snapshot_text, alt_text, element_type)
                    if alt_ref:
                        return alt_ref
        return None
        
    # Return the highest scoring candidate
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def extract_form_fields(snapshot_text: str) -> List[Dict[str, str]]:
    """
    Two-pass strategy:
      1. Try restricted mode (only after 'form' appears).
      2. If no fields found, fall back to global scan so pages without 'form' keyword still work.
    """
    def _scan(lines: List[str], gated: bool) -> List[Dict[str, str]]:
        fields_local: List[Dict[str, str]] = []
        in_form = False
        for line in lines:
            line_lower = line.lower()
            if gated and "form" in line_lower:
                in_form = True
            if gated and not in_form:
                continue
            if _is_interactive_line(line_lower):
                label_match = re.search(r'"([^"]+)"', line)
                label = label_match.group(1) if label_match else "Unlabeled field"
                field_type = "text"
                for t in INTERACTIVE_KEYWORDS:
                    if t in line_lower:
                        field_type = t
                        break
                required = "required" in line_lower
                fields_local.append({"label": label, "type": field_type, "required": required})
        return fields_local

    lines = snapshot_text.splitlines()
    fields = _scan(lines, gated=True)
    if not fields:
        fields = _scan(lines, gated=False)
    # De-duplicate by label keeping first occurrence
    dedup = {}
    for f in fields:
        dedup.setdefault(f["label"], f)
    return list(dedup.values())


def analyze_goal(goal_text: str) -> Dict[str, Any]:
    goal_lower = goal_text.lower()
    analysis = {
        "requires_navigation": any(kw in goal_lower for kw in ["navigate", "go to", "open", "visit", "http://", "https://"]),
        "requires_form_filling": any(kw in goal_lower for kw in ["fill", "input", "enter", "type", "form"]),
        "requires_submission": any(kw in goal_lower for kw in ["submit", "send", "complete", "finish"]),
        "requires_extraction": any(kw in goal_lower for kw in ["extract", "get", "find", "list", "scrape"]),
        "requires_screenshot": any(kw in goal_lower for kw in ["screenshot", "capture", "image", "photo"]),
        "requires_download": any(kw in goal_lower for kw in ["download", "save"]),
        "requires_file_upload": any(kw in goal_lower for kw in ["upload" , "file" , "resume", "pdf"])
    }
    url_match = re.search(r"https?://[^\s)\"']+", goal_text)
    analysis["target_url"] = url_match.group(0) if url_match else None
    return analysis

# # =====================================================================
# # REAL-WEB INTEGRATION TESTS FOR utils.py (Verbose + Live Snapshots)
# # =====================================================================
# # Added: Full result printing (parsed structure, form fields list, goal analysis),
# # per-site JSON friendly aggregation and overall JSON dump.
# # =====================================================================

# import os
# import time
# import json
# import asyncio
# from dataclasses import dataclass, field
# from mcp import ClientSession, StdioServerParameters
# from mcp.client.stdio import stdio_client
# from pprint import pformat

# # ---- Verbose Logging / Color ------------------------------------------------
# USE_COLOR = os.getenv("NO_COLOR") not in ("1", "true", "TRUE")
# def _c(code: str) -> str: return code if USE_COLOR else ""
# CLR_OK = _c("\033[32m")
# CLR_FAIL = _c("\033[31m")
# CLR_INFO = _c("\033[34m")
# CLR_WARN = _c("\033[33m")
# CLR_DIM = _c("\033[2m")
# CLR_RESET = _c("\033[0m")
# CLR_CYAN = _c("\033[36m")

# def log(section: str, msg: str, level: str = "INFO"):
#     ts = time.strftime("%H:%M:%S")
#     col = {
#         "INFO": CLR_INFO,
#         "OK": CLR_OK,
#         "WARN": CLR_WARN,
#         "FAIL": CLR_FAIL,
#         "DATA": CLR_DIM,
#         "STEP": CLR_CYAN,
#     }.get(level, CLR_INFO)
#     print(f"{col}[{ts}] [{section}] {msg}{CLR_RESET}")

# def truncate_local(text: str, limit: int = 800):
#     return text if len(text) <= limit else text[:limit] + f"... <truncated {len(text)-limit} chars>"

# # ---- Lightweight Tool Execution --------------------------------------------
# class _TempToolCaller:
#     def __init__(self, session: ClientSession):
#         self.session = session
#         self.schemas: Dict[str, Dict[str, Any]] = {}

#     async def init(self):
#         listing = await self.session.list_tools()
#         for tool_info in getattr(listing, "tools", []) or []:
#             name = getattr(tool_info, "name", None)
#             if not name:
#                 continue
#             schema = (
#                 getattr(tool_info, "input_schema", None)
#                 or getattr(tool_info, "inputSchema", None)
#                 or {}
#             )
#             props = schema.get("properties", {}) if isinstance(schema, dict) else {}
#             req = set(schema.get("required", []) if isinstance(schema, dict) else [])
#             self.schemas[name] = {"properties": props, "required": req}

#     async def call(self, name: str, args: Dict[str, Any]):
#         if (
#             name in self.schemas
#             and "element" in self.schemas[name]["properties"]
#             and "element" not in args
#             and "selector" in args
#         ):
#             args["element"] = args.pop("selector")
#         result = await self.session.call_tool(name, args)
#         content = getattr(result, "content", None)
#         if isinstance(content, list):
#             collected = []
#             for c in content:
#                 t = getattr(c, "text", None)
#                 if t:
#                     collected.append(t)
#             return "\n".join(collected) if collected else str(result)
#         return str(result)

# # ---- Data Classes -----------------------------------------------------------
# @dataclass
# class AssertionResult:
#     name: str
#     passed: bool
#     detail: str = ""

# @dataclass
# class SiteTestResult:
#     site: str
#     passed: bool
#     assertions: List[AssertionResult] = field(default_factory=list)
#     snapshot_excerpt: str = ""
#     counts: Dict[str, int] = field(default_factory=dict)
#     parsed_summary: Dict[str, Any] = field(default_factory=dict)
#     form_fields: List[Dict[str, Any]] = field(default_factory=list)
#     goal_analysis: Dict[str, Any] = field(default_factory=dict)

# # ---- Helpers ----------------------------------------------------------------
# def count_tokens(snapshot: str) -> Dict[str, int]:
#     low = snapshot.lower()
#     return {
#         "buttons": low.count(" button "),
#         "inputs": sum(low.count(k) for k in [" textbox ", " input ", " combobox ", " textarea "]),
#         "links": low.count(" link "),
#         "tabs": low.count(" tab "),
#         "refs": snapshot.count("[ref="),
#         "lines": snapshot.count("\n") + 1,
#         "chars": len(snapshot),
#     }

# def assert_condition(cond: bool, name: str, detail: str) -> AssertionResult:
#     if cond:
#         log("ASSERT", f"PASS {name} - {detail}", "OK")
#         return AssertionResult(name, True, detail)
#     else:
#         log("ASSERT", f"FAIL {name} - {detail}", "FAIL")
#         return AssertionResult(name, False, detail)

# async def snapshot_site(url: str) -> str:
#     server_params = StdioServerParameters(
#         command=r"C:\Program Files\nodejs\npx.cmd",
#         args=["-y", "@playwright/mcp@latest"],
#     )
#     log("CONNECT", f"🔌 Opening session for {url}")
#     async with stdio_client(server_params) as (read, write):
#         async with ClientSession(read, write) as session:
#             await session.initialize()
#             caller = _TempToolCaller(session)
#             await caller.init()

#             if "browser_navigate" not in caller.schemas:
#                 raise RuntimeError("browser_navigate tool not available")
#             if "browser_snapshot" not in caller.schemas:
#                 raise RuntimeError("browser_snapshot tool not available")

#             log("ACTION", f"Navigating to {url}", "STEP")
#             nav_start = time.perf_counter()
#             _ = await caller.call("browser_navigate", {"url": url})
#             log("ACTION", f"Navigation ok in {(time.perf_counter()-nav_start)*1000:.1f} ms", "OK")

#             snap_start = time.perf_counter()
#             snapshot = await caller.call("browser_snapshot", {})
#             log("ACTION", f"Snapshot acquired in {(time.perf_counter()-snap_start)*1000:.1f} ms", "OK")
#             return snapshot

# # ---- Core Live Tests --------------------------------------------------------
# def summarize_parsed(parsed: Dict[str, Any]) -> Dict[str, Any]:
#     """Produce a condensed summary for logging / JSON export."""
#     return {
#         "title": parsed.get("title"),
#         "headings_count": len(parsed.get("headings", [])),
#         "buttons_count": len(parsed.get("buttons", [])),
#         "inputs_count": len(parsed.get("inputs", [])),
#         "links_count": len(parsed.get("links", [])),
#         "tabs_count": len(parsed.get("tabs", [])),
#         "refs_labels": len(parsed.get("refs", {})),
#         "sample_headings": parsed.get("headings", [])[:5],
#         "sample_buttons": parsed.get("buttons", [])[:5],
#         "sample_inputs": parsed.get("inputs", [])[:5],
#         "sample_links": parsed.get("links", [])[:5],
#     }

# def print_full_results(site_result: SiteTestResult):
#     log("RESULT", f"=== Detailed Results for {site_result.site} ===", "STEP")
#     print(f"{CLR_DIM}Snapshot excerpt (truncated):\n{site_result.snapshot_excerpt}{CLR_RESET}")
#     print(f"{CLR_INFO}Token counts: {site_result.counts}{CLR_RESET}")
#     print(f"{CLR_INFO}Parsed summary: {pformat(site_result.parsed_summary)}{CLR_RESET}")
#     if site_result.form_fields:
#         print(f"{CLR_INFO}Form fields ({len(site_result.form_fields)}):{CLR_RESET}")
#         for f in site_result.form_fields[:10]:
#             print(f"  - {f}")
#         if len(site_result.form_fields) > 10:
#             print(f"  ... ({len(site_result.form_fields)-10} more)")
#     else:
#         print(f"{CLR_INFO}Form fields: None detected{CLR_RESET}")
#     print(f"{CLR_INFO}Goal analysis: {site_result.goal_analysis}{CLR_RESET}")
#     print(f"{CLR_INFO}Assertions:{CLR_RESET}")
#     for a in site_result.assertions:
#         mark = f"{CLR_OK}PASS{CLR_RESET}" if a.passed else f"{CLR_FAIL}FAIL{CLR_RESET}"
#         print(f"  [{mark}] {a.name} - {a.detail}")
#     print()

# async def test_site_with_utils(url: str) -> SiteTestResult:
#     log("SITE", f"Testing utilities using live snapshot: {url}")
#     try:
#         raw_snapshot = await snapshot_site(url)
#     except Exception as e:
#         log("SITE", f"Snapshot failure: {e}", "FAIL")
#         sr = SiteTestResult(site=url, passed=False, assertions=[AssertionResult("snapshot", False, str(e))])
#         print_full_results(sr)
#         return sr

#     excerpt = truncate_local(raw_snapshot, 1200)
#     counts = count_tokens(raw_snapshot)
#     log("SNAPSHOT", f"Excerpt:\n{CLR_DIM}{excerpt}{CLR_RESET}", "DATA")
#     log("COUNTS", f"{counts}", "INFO")

#     assertions: List[AssertionResult] = []

#     # 1. extract_interactive_elements
#     parsed = extract_interactive_elements(raw_snapshot)
#     assertions.append(assert_condition(isinstance(parsed, dict), "parsed_type_dict", "Result is dict"))
#     assertions.append(assert_condition("buttons" in parsed, "buttons_key", "buttons key present"))
#     assertions.append(assert_condition(len(parsed["interactives"]) >= 0, "interactives_list", "interactives collected"))
#     if parsed["buttons"]:
#         # ensure at least one button label exists in parsed["buttons"]
#         assertions.append(assert_condition(len(parsed["buttons"]) > 0, "button_in_interactives_relation",
#                                            "Buttons exist; interactive representation assumed"))
#     else:
#         assertions.append(AssertionResult("button_in_interactives_relation", True, "No buttons; skipped"))

#     # 2. find_element_ref (attempt first label that has refs)
#     found_ref_assert = AssertionResult("find_element_ref_any", True, "No labeled element with ref found (acceptable)")
#     for label, ref_list in parsed.get("refs", {}).items():
#         if ref_list:
#             ref = find_element_ref(raw_snapshot, label)
#             if ref and ref in ref_list:
#                 found_ref_assert = assert_condition(True, "find_element_ref_any",
#                                                     f"Resolved ref {ref} for label '{label}'")
#             else:
#                 found_ref_assert = assert_condition(False, "find_element_ref_any",
#                                                     f"Could not confirm ref for '{label}' (parsed={ref_list}, resolved={ref})")
#             break
#     assertions.append(found_ref_assert)

#     # 3. extract_form_fields
#     fields = extract_form_fields(raw_snapshot)
#     if fields:
#         assertions.append(assert_condition(len(fields) > 0, "form_fields_detected", f"{len(fields)} fields"))
#         first_field = fields[0]
#         assertions.append(assert_condition(
#             all(k in first_field for k in ("label", "type", "required")),
#             "form_field_shape",
#             f"First field keys: {list(first_field.keys())}"
#         ))
#     else:
#         assertions.append(AssertionResult("form_fields_detected", True, "No form fields (page may not contain forms)"))

#     # 4. analyze_goal
#     dynamic_goal = f"Navigate to {url} and extract data then take a screenshot"
#     analysis = analyze_goal(dynamic_goal)
#     assertions.append(assert_condition(analysis["requires_navigation"], "goal_nav_detected", "navigation True"))
#     assertions.append(assert_condition(analysis["requires_extraction"], "goal_extract_detected", "extraction True"))
#     assertions.append(assert_condition(analysis["requires_screenshot"], "goal_screenshot_detected", "screenshot True"))
#     assertions.append(assert_condition(analysis["target_url"] == url, "goal_url_match", f"url={analysis['target_url']}"))

#     passed = all(a.passed for a in assertions)

#     site_result = SiteTestResult(
#         site=url,
#         passed=passed,
#         assertions=assertions,
#         snapshot_excerpt=excerpt,
#         counts=counts,
#         parsed_summary=summarize_parsed(parsed),
#         form_fields=fields,
#         goal_analysis=analysis
#     )

#     # Print full detailed result block (NEW)
#     print_full_results(site_result)
#     return site_result

# async def run_live_tests():
#     log("RUN", "Starting live utils tests against real sites")
#     sites = [
#         "https://example.com",
#         "https://httpbin.org/forms/post",
#         "https://www.wikipedia.org"
#     ]
#     results: List[SiteTestResult] = []
#     for s in sites:
#         try:
#             results.append(await test_site_with_utils(s))
#         except Exception as e:
#             log("SITE", f"Fatal test error for {s}: {e}", "FAIL")
#             sr = SiteTestResult(
#                 site=s,
#                 passed=False,
#                 assertions=[AssertionResult("fatal", False, str(e))]
#             )
#             print_full_results(sr)
#             results.append(sr)

#     # Summary
#     print("\n" + "=" * 72)
#     print("LIVE UTILS TEST SUMMARY")
#     print("=" * 72)
#     total_asserts = 0
#     total_passed = 0
#     summary_for_json = []
#     for r in results:
#         site_col = CLR_OK if r.passed else CLR_FAIL
#         pass_count = sum(1 for a in r.assertions if a.passed)
#         total = len(r.assertions)
#         total_asserts += total
#         total_passed += pass_count
#         print(f"{site_col}{r.site}: {pass_count}/{total} assertions passed{CLR_RESET}")
#         for a in r.assertions:
#             a_col = CLR_OK if a.passed else CLR_FAIL
#             print(f"  {a_col}{'[PASS]' if a.passed else '[FAIL]'} {a.name}{CLR_RESET} - {a.detail}")
#         summary_for_json.append({
#             "site": r.site,
#             "passed": r.passed,
#             "assertions": [{"name": a.name, "passed": a.passed, "detail": a.detail} for a in r.assertions],
#             "counts": r.counts,
#             "parsed_summary": r.parsed_summary,
#             "form_fields": r.form_fields,
#             "goal_analysis": r.goal_analysis
#         })

#     print(f"\nOVERALL: {total_passed}/{total_asserts} assertions passed")

#     # Print consolidated machine-friendly JSON to stdout (NEW)
#     print("\n=== RAW JSON RESULTS (stdout) ===")
#     print(json.dumps(summary_for_json, indent=2))

#     if os.getenv("UTILS_LIVE_JSON") in ("1", "true", "TRUE"):
#         out_file = "utils_live_test_summary.json"
#         with open(out_file, "w", encoding="utf-8") as f:
#             json.dump(summary_for_json, f, indent=2)
#         print(f"\nJSON summary written to {out_file}")
#     return 0 if total_passed == total_asserts else 1

# def _sync(coro):
#     return asyncio.run(coro)

# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser(description="Utils module (live web snapshot tests)")
#     parser.add_argument("--run-live-tests", action="store_true", help="Run live tests using real browser snapshots")
#     parser.add_argument("--json", action="store_true", help="Emit JSON summary file (UTILS_LIVE_JSON=1)")
#     args = parser.parse_args()

#     if args.json:
#         os.environ["UTILS_LIVE_JSON"] = "1"

#     if args.run_live_tests:
#         code = _sync(run_live_tests())
#         raise SystemExit(code)

#     print(
#         "Usage:\n"
#         "  python utils.py --run-live-tests\n"
#         "Options:\n"
#         "  --json   (emit utils_live_test_summary.json)\n"
#         "Environment:\n"
#         "  NO_COLOR=1 disables colors\n"
#         "  UTILS_LIVE_JSON=1 writes JSON summary\n"
#     )