"""Enhanced utility functions for web automation agent."""
import re
from typing import Dict, Any, List, Optional

# ---------------- Core Utilities ---------------- #

def truncate_text(text: str, limit: int = 1000) -> str:
    if not text or len(text) <= limit:
        return text
    return text[:limit] + f"... [truncated {len(text) - limit} characters]"


INTERACTIVE_KEYWORDS = ("textbox", "input", "combobox", "textarea", "checkbox", "dropdown", "radio", "select", "button", "radiogroup")

def _is_interactive_line(line_lower: str) -> bool:
    return any(k in line_lower for k in INTERACTIVE_KEYWORDS)


def extract_interactive_elements(snapshot_text: str) -> Dict[str, Any]:
    print("\nSNAPSHOT TEXT", snapshot_text, "\n")
    lines = snapshot_text.splitlines()
    result = {
        "title": None,
        "headings": [],
        "buttons": [],
        "inputs": [],
        "links": [],
        "tabs": [],
        "refs": {},
        "dropdowns":[],
        "interactives": [],
        "all_clickables": [],
        "form_sections": [],
        "file_uploads": [],
        "radio_groups": [],
        "comboboxes": [],
        "date_fields": []
    }
    
    ref_pattern = re.compile(r"\[ref=((?:f\d+)?e\d+)\]")  # Updated to handle iframe refs like f1e2
    quoted_pattern = re.compile(r'"([^"]+)"')
    icon_pattern = re.compile(r'<svg[^>]*>.*?</svg>|<i[^>]*>.*?</i>|<img[^>]*>', re.DOTALL)
    
    # Enhanced context mapping - collect more comprehensive context
    context_map = {}
    section_context = ""  # Track current section/heading
    
    for i, line in enumerate(lines):
        line_lower = line.lower()
        line_stripped = line.strip()
        
        # Track section headings for context
        if "heading" in line_lower:
            heading_match = quoted_pattern.search(line)
            if heading_match:
                section_context = heading_match.group(1)
                result["headings"].append(section_context)
        
        refs = ref_pattern.findall(line)
        
        if refs:
            for ref in refs:
                # Collect more comprehensive context
                context_lines = []
                
                # Add current section context
                if section_context:
                    context_lines.append(f"Section: {section_context}")
                
                # Look backwards for context (up to 5 lines)
                for j in range(max(0, i-5), i):
                    prev_line = lines[j].strip()
                    if prev_line and not ref_pattern.search(prev_line):
                        # Extract meaningful text
                        text_content = re.sub(r'^\s*-\s*', '', prev_line)
                        text_content = re.sub(r':\s*$', '', text_content)
                        text_content = re.sub(r'^\s*generic.*?:\s*', '', text_content)
                        
                        # Skip very generic content
                        skip_terms = ['generic', 'text:', 'img', 'button:', 'textbox:', 'link:']
                        if (len(text_content) > 2 and 
                            not any(term in text_content.lower() for term in skip_terms) and
                            not text_content.startswith('- ') and
                            text_content not in context_lines):
                            context_lines.append(text_content)
                
                # Look forwards for context (up to 3 lines)
                for j in range(i+1, min(len(lines), i+4)):
                    next_line = lines[j].strip()
                    if next_line and not ref_pattern.search(next_line):
                        text_content = re.sub(r'^\s*-\s*', '', next_line)
                        text_content = re.sub(r':\s*$', '', text_content)
                        text_content = re.sub(r'^\s*generic.*?:\s*', '', text_content)
                        
                        skip_terms = ['generic', 'text:', 'img', 'button:', 'textbox:', 'link:']
                        if (len(text_content) > 2 and 
                            not any(term in text_content.lower() for term in skip_terms) and
                            not text_content.startswith('- ') and
                            text_content not in context_lines):
                            context_lines.append(text_content)
                
                context_map[ref] = context_lines
    
    # Enhanced element extraction
    file_keywords = ["upload", "file", "resume", "cv", "document", "pdf", "browse files", "drag and drop"]
    date_keywords = ["date", "calendar", "choose date"]
    
    for i, line in enumerate(lines):
        line_lower = line.lower()
        line_stripped = line.strip()
        
        # Extract title
        if result["title"] is None and "page title:" in line_lower:
            result["title"] = line.split(":", 1)[-1].strip()
            continue
        
        # Handle special cases - comboboxes without quotes
        if "combobox" in line_lower and not quoted_pattern.search(line):
            refs = ref_pattern.findall(line)
            if refs:
                # Look for context in previous lines for combobox labels
                context_lines = context_map.get(refs[0], [])
                for context in context_lines:
                    if len(context) > 5 and not any(skip in context.lower() for skip in ['generic', 'text:', 'section:']):
                        result["comboboxes"].append(context.strip())
                        result["interactives"].append(f"combobox: {context.strip()}")
                        if context.strip() not in result["refs"]:
                            result["refs"][context.strip()] = refs
                        break
        
        # Enhanced interactive element detection - handle all element types from snapshot
        interactive_types = {
            "button": ["button"],
            "textbox": ["textbox", "input"],
            "combobox": ["combobox", "select"],
            "textarea": ["textarea"],
            "checkbox": ["checkbox"],
            "radio": ["radio"],
            "radiogroup": ["radiogroup"],
            "link": ["link"],
            "tab": ["tab"],
            "spinbutton": ["spinbutton"],
            "group": ["group"],
            "option": ["option"],  # Handle dropdown options
            "list": ["list"],      # Handle file upload lists
            "img": ["img"]         # Handle clickable images/icons
        }
        
        detected_type = None
        for element_type, keywords in interactive_types.items():
            if any(f' {kw} ' in f' {line_lower} ' for kw in keywords):
                detected_type = element_type
                break
        
        if detected_type:
            # Extract labels from quotes
            labels = quoted_pattern.findall(line)
            refs = ref_pattern.findall(line)
            
            # Enhanced label extraction for elements without quoted labels
            if not labels and refs:
                ref = refs[0]
                context_lines = context_map.get(ref, [])
                
                # Try different label extraction strategies
                potential_labels = []
                
                # Strategy 1: Extract from current line patterns
                if "Browse Files" in line:
                    potential_labels.append("Browse Files")
                elif "Add Row" in line:
                    potential_labels.append("Add Row")
                elif "Choose Date" in line:
                    potential_labels.append("Choose Date")
                elif "Submit" in line:
                    potential_labels.append("Submit")
                elif "Clear" in line:
                    potential_labels.append("Clear")
                
                # Strategy 2: Use context lines
                for context in context_lines:
                    clean_context = re.sub(r'^(Section:|text:\s*)', '', context).strip()
                    clean_context = re.sub(r'[*"]+', '', clean_context).strip()
                    
                    if (len(clean_context) > 1 and 
                        clean_context.lower() not in ['generic', 'text', 'img', 'please select'] and
                        not clean_context.startswith('- ')):
                        potential_labels.append(clean_context)
                
                # Strategy 3: Extract from element attributes
                if not potential_labels:
                    # Look for meaningful text in the line itself
                    text_parts = line.split()
                    for part in text_parts:
                        if (len(part) > 2 and 
                            part not in ['generic', 'ref', 'cursor'] and
                            not part.startswith('[') and 
                            not part.endswith(']')):
                            potential_labels.append(part)
                
                # Select best label
                if potential_labels:
                    # Prioritize certain types of labels
                    best_label = potential_labels[0]
                    for label in potential_labels:
                        label_lower = label.lower()
                        # Prefer descriptive labels
                        if any(term in label_lower for term in ['upload', 'file', 'browse', 'submit', 'date', 'name', 'email', 'address']):
                            best_label = label
                            break
                    labels = [best_label]
            
            # Process each label
            for label in labels:
                if label:
                    label = label.strip().replace('"', '')
                    if not label or label.lower() in ['generic', 'text']:
                        continue
                    
                    # Add to appropriate categories
                    element_info = {
                        "label": label,
                        "type": detected_type,
                        "ref": refs[0] if refs else None,
                        "line": line_stripped
                    }
                    
                    # Add to appropriate categories with enhanced detection
                    if detected_type in ["button"] and label not in result["buttons"]:
                        result["buttons"].append(label)
                    elif detected_type in ["textbox", "input", "textarea", "spinbutton"] and label not in result["inputs"]:
                        result["inputs"].append(label)
                    elif detected_type in ["combobox", "select"] and label not in result["comboboxes"]:
                        result["comboboxes"].append(label)
                    elif detected_type == "link" and label not in result["links"]:
                        result["links"].append(label)
                    elif detected_type == "tab" and label not in result["tabs"]:
                        result["tabs"].append(label)
                    elif detected_type in ["radiogroup", "radio", "group"] and label not in result["radio_groups"]:
                        result["radio_groups"].append(label)
                    elif detected_type == "checkbox":
                        checkbox_info = f"checkbox: {label}"
                        if checkbox_info not in result["interactives"]:
                            result["interactives"].append(checkbox_info)
                    elif detected_type == "list" and "uploaded files" in label.lower():
                        list_info = f"file_list: {label}"
                        if list_info not in result["file_uploads"]:
                            result["file_uploads"].append(list_info)
                    
                    # Special handling for file uploads
                    if any(kw in line_lower for kw in file_keywords):
                        file_info = f"file_upload: {label}"
                        if file_info not in result["file_uploads"]:
                            result["file_uploads"].append(file_info)
                    
                    # Special handling for date fields
                    if any(kw in line_lower for kw in date_keywords):
                        date_info = f"date_field: {label}"
                        if date_info not in result["date_fields"]:
                            result["date_fields"].append(date_info)
                    
                    # Add to general collections
                    interactive_desc = f"{detected_type}: {label}"
                    if interactive_desc not in result["interactives"]:
                        result["interactives"].append(interactive_desc)
                    
                    if detected_type in ["button", "tab", "link"] and label not in result["all_clickables"]:
                        result["all_clickables"].append(label)
                    
                    # Store ref associations
                    if refs:
                        for ref in refs:
                            if label not in result["refs"]:
                                result["refs"][label] = []
                            if ref not in result["refs"][label]:
                                result["refs"][label].append(ref)
    
    # Additional processing for iframe content and nested elements
    iframe_count = 0
    for line in lines:
        if "iframe" in line.lower():
            iframe_count += 1
            result["form_sections"].append(f"iframe_section_{iframe_count}")
    
    # Additional comprehensive processing for complex forms
    # Extract dropdown/combobox options and radiogroup questions
    for i, line in enumerate(lines):
        line_lower = line.lower()
        
        # Handle radiogroups with skill ratings
        if "radiogroup" in line_lower:
            labels = quoted_pattern.findall(line)
            refs = ref_pattern.findall(line)
            if labels and refs:
                skill_name = labels[0]
                if skill_name not in result["radio_groups"]:
                    result["radio_groups"].append(skill_name)
                    result["interactives"].append(f"skill_rating: {skill_name}")
                    result["refs"][skill_name] = refs
        
        # Handle group elements (Yes/No questions)
        elif "group" in line_lower:
            labels = quoted_pattern.findall(line)
            refs = ref_pattern.findall(line)
            if labels and refs:
                question = labels[0]
                if question not in result["radio_groups"]:
                    result["radio_groups"].append(question)
                    result["interactives"].append(f"yes_no_question: {question}")
                    result["refs"][question] = refs
        
        # Handle standalone combobox questions without direct quotes
        elif "combobox" in line_lower and quoted_pattern.search(line):
            labels = quoted_pattern.findall(line)
            refs = ref_pattern.findall(line)
            if labels and refs:
                question = labels[0]
                if question not in result["comboboxes"]:
                    result["comboboxes"].append(question)
                    result["interactives"].append(f"dropdown: {question}")
                    result["refs"][question] = refs
        
        # Extract individual radio button options within groups
        elif "radio" in line_lower and not "radiogroup" in line_lower:
            labels = quoted_pattern.findall(line)
            refs = ref_pattern.findall(line)
            if labels and refs:
                option = labels[0]
                # Look for parent context
                context_lines = context_map.get(refs[0], [])
                parent_question = None
                for context in context_lines:
                    if "?" in context and len(context) > 10:
                        parent_question = context.strip('?"')
                        break
                
                option_info = f"radio_option: {option}"
                if parent_question:
                    option_info += f" (for: {parent_question})"
                
                if option_info not in result["interactives"]:
                    result["interactives"].append(option_info)
                    result["refs"][f"{option} (radio)"] = refs
        
        # Handle checkbox elements specifically
        elif "checkbox" in line_lower:
            labels = quoted_pattern.findall(line)
            refs = ref_pattern.findall(line)
            if labels and refs:
                checkbox_label = labels[0]
                checkbox_info = f"checkbox: {checkbox_label}"
                if checkbox_info not in result["interactives"]:
                    result["interactives"].append(checkbox_info)
                    result["refs"][checkbox_label] = refs
                    result["all_clickables"].append(checkbox_label)
    
    # Extract questions/labels that appear before form elements
    question_candidates = []
    for i, line in enumerate(lines):
        line_stripped = line.strip()
        # Look for question-like patterns
        if (line_stripped.endswith('?') and 
            len(line_stripped) > 10 and 
            not any(kw in line_stripped.lower() for kw in INTERACTIVE_KEYWORDS)):
            
            # Check if next few lines contain interactive elements
            has_following_interactive = False
            for j in range(i+1, min(len(lines), i+5)):
                if _is_interactive_line(lines[j].lower()):
                    has_following_interactive = True
                    break
            
            if has_following_interactive:
                clean_question = re.sub(r'^- ', '', line_stripped)
                clean_question = re.sub(r'generic.*?:', '', clean_question).strip()
                if clean_question and clean_question not in question_candidates:
                    question_candidates.append(clean_question)
    
    # Add question candidates to appropriate categories if not already captured
    for question in question_candidates:
        if not any(question in existing for existing in 
                  result["comboboxes"] + result["radio_groups"] + result["inputs"]):
            result["interactives"].append(f"question: {question}")
    
    return result


def find_element_ref(snapshot_text: str, element_text: str, element_type: str = None) -> Optional[str]:
    """
    Enhanced element finding with better pattern matching and iframe support.
    """
    lines = snapshot_text.splitlines()
    element_lower = element_text.lower()
    
    candidates = []
    
    # Enhanced reference pattern to handle iframe refs
    ref_pattern = re.compile(r"\[ref=((?:f\d+)?e\d+)\]")
    
    for idx, line in enumerate(lines):
        line_lower = line.lower()
        
        # Enhanced interactive detection
        interactive_keywords = [
            "button", "link", "tab", "textbox", "input", "clickable", "combobox", 
            "textarea", "checkbox", "radio", "select", "spinbutton", "radiogroup"
        ]
        
        if not any(kw in line_lower for kw in interactive_keywords):
            continue
            
        score = 0
        
        # Enhanced matching strategies
        if f'"{element_text}"' in line:
            score = 100  # Exact quoted match
        elif element_lower in line_lower:
            score = 80   # Contains match
        elif re.search(rf'\b{re.escape(element_lower)}\b', line_lower):
            score = 70   # Word boundary match
        else:
            # Fuzzy matching for common variations
            element_words = element_lower.split()
            line_words = line_lower.split()
            common_words = set(element_words) & set(line_words)
            if common_words and len(common_words) >= len(element_words) * 0.6:
                score = 60
            elif any(word in line_lower for word in element_words if len(word) > 2):
                score = 40
            else:
                continue
                
        # Type matching bonus
        if element_type and element_type.lower() in line_lower:
            score += 20
            
        # Special bonuses for specific patterns
        if "browse files" in element_lower and "browse files" in line_lower:
            score += 30
        if "choose date" in element_lower and "choose date" in line_lower:
            score += 30
        if "add row" in element_lower and "add row" in line_lower:
            score += 30
            
        # Extract reference
        ref_match = ref_pattern.search(line)
        if ref_match:
            ref = ref_match.group(1)
            candidates.append((score, ref, line, idx))
            
            # Also check nearby lines for context
            for offset in [-1, 1]:
                nearby_idx = idx + offset
                if 0 <= nearby_idx < len(lines):
                    nearby_ref_match = ref_pattern.search(lines[nearby_idx])
                    if nearby_ref_match:
                        nearby_ref = nearby_ref_match.group(1)
                        candidates.append((score - 10, nearby_ref, lines[nearby_idx], nearby_idx))
    
    if not candidates:
        # Try alternative matching strategies
        alternative_strategies = {
            "apply here": ["apply", "submit", "continue", "next", "start application"],
            "browse files": ["upload", "file", "browse", "choose file"],
            "choose date": ["date", "calendar", "pick date"],
            "submit": ["send", "complete", "finish", "apply"],
        }
        
        for alt_pattern, alternatives in alternative_strategies.items():
            if alt_pattern in element_lower:
                for alt_text in alternatives:
                    if alt_text != element_text:
                        alt_ref = find_element_ref(snapshot_text, alt_text, element_type)
                        if alt_ref:
                            return alt_ref
        return None
        
    # Return the highest scoring candidate
    candidates.sort(key=lambda x: (x[0], -x[3]), reverse=True)  # Score desc, line number asc
    return candidates[0][1]


def extract_form_fields(snapshot_text: str) -> List[Dict[str, str]]:
    """
    Enhanced form field extraction with better detection and iframe support.
    """
    def _scan(lines: List[str], gated: bool) -> List[Dict[str, str]]:
        fields_local: List[Dict[str, str]] = []
        in_form = False
        section_context = ""
        
        for i, line in enumerate(lines):
            line_lower = line.lower()
            
            # Track sections for better context
            if "heading" in line_lower:
                heading_match = re.search(r'"([^"]+)"', line)
                if heading_match:
                    section_context = heading_match.group(1)
            
            if gated and ("form" in line_lower or "personal information" in line_lower or "position information" in line_lower):
                in_form = True
            if gated and not in_form:
                continue
                
            if _is_interactive_line(line_lower):
                # Enhanced label extraction
                label_match = re.search(r'"([^"]+)"', line)
                label = label_match.group(1) if label_match else "Unlabeled field"
                
                # If no label found, try to get from context
                if label == "Unlabeled field" and section_context:
                    # Look at nearby lines for context
                    for j in range(max(0, i-3), min(len(lines), i+3)):
                        context_line = lines[j].strip()
                        if (context_line and 
                            not re.search(r'\[ref=', context_line) and
                            not any(kw in context_line.lower() for kw in INTERACTIVE_KEYWORDS)):
                            # Clean up context
                            clean_context = re.sub(r'^- ', '', context_line)
                            clean_context = re.sub(r':$', '', clean_context)
                            if len(clean_context) > 1 and clean_context != "generic":
                                label = clean_context
                                break
                
                # Determine field type with enhanced detection
                field_type = "text"
                type_mappings = {
                    "textbox": "text",
                    "input": "text", 
                    "textarea": "textarea",
                    "combobox": "select",
                    "select": "select",
                    "checkbox": "checkbox",
                    "radio": "radio",
                    "radiogroup": "radio",
                    "spinbutton": "number",
                    "button": "button"
                }
                
                for keyword, mapped_type in type_mappings.items():
                    if keyword in line_lower:
                        field_type = mapped_type
                        break
                
                # Special field type detection
                if any(kw in label.lower() for kw in ["email", "@"]):
                    field_type = "email"
                elif any(kw in label.lower() for kw in ["phone", "tel"]):
                    field_type = "tel"
                elif any(kw in label.lower() for kw in ["date", "calendar"]):
                    field_type = "date"
                elif any(kw in label.lower() for kw in ["upload", "file", "cv", "resume"]):
                    field_type = "file"
                elif any(kw in label.lower() for kw in ["password", "pass"]):
                    field_type = "password"
                
                required = "required" in line_lower or "*" in line
                
                field_info = {
                    "label": label,
                    "type": field_type,
                    "required": required,
                    "section": section_context
                }
                
                fields_local.append(field_info)
        
        return fields_local

    lines = snapshot_text.splitlines()
    fields = _scan(lines, gated=True)
    if not fields:
        fields = _scan(lines, gated=False)
    
    # Enhanced deduplication - keep most informative version
    dedup = {}
    for f in fields:
        label = f["label"]
        if label not in dedup or (f.get("section") and not dedup[label].get("section")):
            dedup[label] = f
    
    return list(dedup.values())


def analyze_goal(goal_text: str) -> Dict[str, Any]:
    """Enhanced goal analysis with more comprehensive detection."""
    goal_lower = goal_text.lower()
    analysis = {
        "requires_navigation": any(kw in goal_lower for kw in ["navigate", "go to", "open", "visit", "http://", "https://", "browse to"]),
        "requires_form_filling": any(kw in goal_lower for kw in ["fill", "input", "enter", "type", "form", "complete form", "application"]),
        "requires_submission": any(kw in goal_lower for kw in ["submit", "send", "complete", "finish", "apply", "save"]),
        "requires_extraction": any(kw in goal_lower for kw in ["extract", "get", "find", "list", "scrape", "collect", "gather"]),
        "requires_screenshot": any(kw in goal_lower for kw in ["screenshot", "capture", "image", "photo", "snap"]),
        "requires_download": any(kw in goal_lower for kw in ["download", "save", "export"]),
        "requires_file_upload": any(kw in goal_lower for kw in ["upload", "file", "resume", "pdf", "document", "cv", "attach"]),
        "requires_selection": any(kw in goal_lower for kw in ["select", "choose", "pick", "option", "dropdown"]),
        "requires_authentication": any(kw in goal_lower for kw in ["login", "sign in", "authenticate", "password", "username"])
    }
    
    # Enhanced URL extraction
    url_patterns = [
        r"https?://[^\s)\"']+",
        r"www\.[^\s)\"']+",
        r"[a-zA-Z0-9-]+\.[a-zA-Z]{2,}[^\s)\"']*"
    ]
    
    for pattern in url_patterns:
        url_match = re.search(pattern, goal_text)
        if url_match:
            url = url_match.group(0)
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url
            analysis["target_url"] = url
            break
    else:
        analysis["target_url"] = None
    
    return analysis