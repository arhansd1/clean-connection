"""Enhanced utility functions for web automation agent (updated with combobox options
and robust handling for non-string snapshot lines).
"""
import re
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

# ---------------- Helpers ---------------- #

def _to_str(x: Any) -> str:
    """Safely coerce various node types to a stable string for parsing.
    If input is already a string, return it; otherwise return repr(x).
    """
    return x if isinstance(x, str) else repr(x)

# ---------------- Enhanced Form Field Parser ---------------- #

@dataclass
class FormField:
    """Structured form field representation."""
    element_type: str
    label: str
    text: str
    ref: str
    field_type: Optional[str] = None
    options: Optional[List[str]] = None
    required: bool = False
    
    def __str__(self):
        """Clean string representation."""
        base = f"{self.element_type.title()} field {{Label: '{self.label}'"
        if self.text and self.text != self.label:
            base += f", Text: '{self.text}'"
        if self.field_type:
            base += f", Type: '{self.field_type}'"
        if self.ref:
            base += f", Ref: '{self.ref}'"
        if self.options:
            base += f", Options: [{', '.join(self.options)}]"
        if self.required:
            base += ", Required: True"
        base += "}"
        return base


class FormFieldParser:
    """Enhanced form field parser with comprehensive error handling."""
    
    def __init__(self):
        self.ref_pattern = re.compile(r"\[ref=((?:f\d+)?e\d+)\]")
        self.quoted_pattern = re.compile(r'"([^\"]+)"')
        self.options_pattern = re.compile(r"options:\[([^\]]+)\]")
        
        # Field type mappings
        self.element_type_mapping = {
            "textbox": "text",
            "input": "text",
            "textarea": "textarea", 
            "combobox": "dropdown",
            "select": "dropdown",
            "skill": "skill_rating",
            "yes/no": "yes_no",
            "checkbox": "checkbox",
            "radio": "radio",
            "radiogroup": "radio_group",
            "spinbutton": "number",
            "button": "button",
            "link": "link",
            "tab": "tab"
        }
        
        # Enhanced field type detection patterns
        self.field_type_patterns = {
            "email": ["email", "@", "e-mail"],
            "phone": ["phone", "tel", "mobile", "telephone"],
            "date": ["date", "calendar", "choose date"],
            "file": ["upload", "file", "cv", "resume", "browse files"],
            "password": ["password", "pass"],
            "number": ["number", "amount", "pay", "salary"],
            "url": ["url", "website", "link"],
            "search": ["search"]
        }
    
    def safe_get_list_item(self, items: List[str], index: int, default: str = "") -> str:
        """Safely get list item with bounds checking."""
        try:
            return items[index] if 0 <= index < len(items) else default
        except (IndexError, TypeError):
            return default
    
    def safe_split(self, text: str, delimiter: str, expected_parts: int) -> List[str]:
        """Safely split text and pad with empty strings if needed."""
        try:
            parts = text.split(delimiter, expected_parts - 1)
            # Pad with empty strings if we don't have enough parts
            while len(parts) < expected_parts:
                parts.append("")
            return parts
        except (AttributeError, TypeError):
            return [""] * expected_parts
    
    # In parse_form_fields_enhanced, update the options extraction:
    def extract_options(self, field_line: str, ref: str) -> List[str]:
        """Extract options specific to a ref"""
        # First try to get options from ref-specific storage
        if hasattr(self, 'page_state') and f"{ref}_options" in self.page_state.get("refs", {}):
            return self.page_state["refs"][f"{ref}_options"]
        
        # Fallback to inline extraction
        options_match = self.options_pattern.search(field_line)
        if options_match:
            options_str = options_match.group(1)
            options = [opt.strip() for opt in options_str.split(',')]
            return [opt for opt in options if opt]
        return []

    

    def extract_field_info(self, field_line: str) -> Tuple[str, str, str, str, List[str]]:
        """Extract field information with robust error handling."""
        try:
            # Ensure we operate on a string
            field_line = _to_str(field_line)

            # Remove leading dash and whitespace
            cleaned_line = field_line.strip().lstrip('- ')
            
            options = self.extract_options(field_line)
            # Split on the first comma to separate element type from rest
            parts = self.safe_split(cleaned_line, ',', 2)
            element_type = self.safe_get_list_item(parts, 0).strip()
            remainder = self.safe_get_list_item(parts, 1).strip()
            
            # Extract label (everything before the ref)
            # Accept patterns like: combobox "Label" [ref=e59]: or generic [ref=e6]: Label
            label = ""
            ref = ""
            # Try quoted label first
            qm = self.quoted_pattern.search(field_line)
            if qm:
                label = qm.group(1).strip()
            else:
                # If no quoted label, try remainder before [ref= or options:
                if 'ref:' in remainder:
                    label_part = remainder.split('ref:', 1)[0]
                    label = re.sub(r':\s*$', '', label_part).strip()
                elif 'options:' in remainder:
                    label_part = remainder.split('options:', 1)[0]  
                    label = re.sub(r':\s*$', '', label_part).strip()
                else:
                    # Fallback to whole cleaned_line minus element_type prefix
                    label = cleaned_line[len(element_type):].strip().strip(':').strip()
                    # Remove ref and options parts
                    if 'ref:' in label:
                        label = label.split('ref:', 1)[0].strip()
                    if 'options:' in label:
                        label = label.split('options:', 1)[0].strip() 

            # Extract ref - look for ref: pattern
            if 'ref:' in field_line:
                ref_part = field_line.split('ref:', 1)[1]
                # Extract until next comma or end
                ref = ref_part.split(',')[0].strip()
            
            # Extract field type if present (type:xxx pattern)
            field_type = ""
            type_match = re.search(r'type:(\w+)', field_line)
            if type_match:
                field_type = type_match.group(1)
            
            return element_type, label, ref, field_type , options
            
        except Exception:
            # Return safe defaults if parsing fails
            return "unknown", _to_str(field_line).strip(), "", "" , []
    
    def detect_field_type(self, label: str, element_type: str, existing_type: str = "") -> str:
        """Detect field type based on label content and element type."""
        if existing_type:
            return existing_type
            
        label_lower = _to_str(label).lower()
        element_lower = _to_str(element_type).lower()
        
        # Check against field type patterns
        for field_type, patterns in self.field_type_patterns.items():
            if any(pattern in label_lower for pattern in patterns):
                return field_type
        
        # Default based on element type
        # return self.element_type_mapping.get(_to_str(element_type).lower(), "text")
        return self.element_type_mapping.get(element_lower, "text")
    
    def extract_radio_options(self, field_lines: List[str], start_index: int) -> Tuple[List[str], int]:
        """Extract radio button options that follow a radio group or group question."""
        options = []
        idx = start_index + 1
        while idx < len(field_lines):
            line = _to_str(field_lines[idx]).strip()
            # Accept patterns that contain 'radio' and a quoted label
            if 'radio' in line.lower() and self.quoted_pattern.search(line):
                qm = self.quoted_pattern.search(line)
                if qm:
                    options.append(qm.group(1).strip())
                idx += 1
                continue
            # In some snapshots options are written as '- option "Text"'
            if line.lower().startswith('- option') and self.quoted_pattern.search(line):
                qm = self.quoted_pattern.search(line)
                options.append(qm.group(1).strip())
                idx += 1
                continue
            break
        return options, idx - 1
    
    def extract_combobox_options(self, field_lines: List[str], start_index: int) -> Tuple[List[str], int]:
        """Extract combobox/select option lines that follow a combobox."""
        options = []
        idx = start_index + 1
        while idx < len(field_lines):
            line = _to_str(field_lines[idx]).strip()
            # Look for '- option "..."' or lines containing 'option "..."'
            if 'option' in line.lower() and self.quoted_pattern.search(line):
                qm = self.quoted_pattern.search(line)
                if qm:
                    opt = qm.group(1).strip()
                    # strip trailing [selected] or other annotations
                    opt = re.sub(r'\s*\[.*\]$', '', opt).strip()
                    if opt:
                        options.append(opt)
                idx += 1
                continue
            # Some combobox option lists are indented as ' - option "xxx"'
            if line.startswith('-') and self.quoted_pattern.search(line) and ('option' in line.lower() or 'please select' in line.lower()):
                qm = self.quoted_pattern.search(line)
                opt = qm.group(1).strip()
                opt = re.sub(r'\s*\[.*\]$', '', opt).strip()
                if opt:
                    options.append(opt)
                idx += 1
                continue
            break
        return options, idx - 1
# ---------------- Core Utilities ---------------- #

def truncate_text(text: str, limit: int = 1000) -> str:
    if not text or len(text) <= limit:
        return text
    return text[:limit] + f"... [truncated {len(text) - limit} characters]"


INTERACTIVE_KEYWORDS = ("textbox", "input", "combobox", "textarea", "checkbox", "dropdown", "radio", "select", "button", "radiogroup")

def _is_interactive_line(line_lower: str) -> bool:
    return any(k in line_lower for k in INTERACTIVE_KEYWORDS)


def _normalize_snapshot_lines(snapshot_text: Any) -> List[str]:
    """Return a list of string lines from either a string snapshot or a list-like snapshot.
    This ensures downstream loops always receive strings.
    """
    if isinstance(snapshot_text, str):
        raw_lines = snapshot_text.splitlines()
    elif isinstance(snapshot_text, list):
        raw_lines = snapshot_text
    else:
        raw_lines = [repr(snapshot_text)]
    return [_to_str(l) for l in raw_lines]


def extract_interactive_elements(snapshot_text: Any) -> Dict[str, Any]:
    print("\nSNAPSHOT TEXT :)", snapshot_text, "\n")
    lines = _normalize_snapshot_lines(snapshot_text)
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
        "date_fields": [],
        "checkboxes": []
    }
    
    ref_pattern = re.compile(r"\[ref=((?:f\d+)?e\d+)\]")  # Updated to handle iframe refs like f1e2
    quoted_pattern = re.compile(r'"([^\"]+)"')
    icon_pattern = re.compile(r'<svg[^>]*>.*?</svg>|<i[^>]*>.*?</i>|<img[^>]*>', re.DOTALL)
    
    # Enhanced context mapping - collect more comprehensive context
    context_map = {}
    section_context = ""  # Track current section/heading
    
    for i, raw_line in enumerate(lines):
        line = _to_str(raw_line)
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
                    prev_line = _to_str(lines[j]).strip()
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
                    next_line = _to_str(lines[j]).strip()
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
    
    for i, raw_line in enumerate(lines):
        line = _to_str(raw_line)
        line_lower = line.lower()
        line_stripped = line.strip()
        
        # Extract title
        if result["title"] is None and "page title:" in line_lower:
            result["title"] = line.split(":", 1)[-1].strip()
            continue
        
        # Enhanced file upload detection - USE REFS INSTEAD OF LABELS
        if any(kw in line_lower for kw in file_keywords):
            refs = ref_pattern.findall(line)
            if refs:
                # Create unique identifier using ref instead of label
                for ref in refs:
                    file_id = f"file_upload_{ref}"
                    if file_id not in result["file_uploads"]:
                        result["file_uploads"].append(file_id)
                        result["interactives"].append(f"file_upload: {file_id}")
                        result["all_clickables"].append(file_id)
                        result["refs"][file_id] = [ref]

        # Enhanced checkbox detection - look for iframe checkbox patterns
        if "checkbox" in line_lower:
            # Extract checkbox label from quoted text or context
            labels = quoted_pattern.findall(line)
            refs = ref_pattern.findall(line)
            
            if refs:
                checkbox_label = "Checkbox"
                if labels:
                    checkbox_label = labels[0]
                else:
                    # Look for context in surrounding lines
                    context_lines = context_map.get(refs[0], [])
                    for context in context_lines:
                        if "agree" in context.lower() or "terms" in context.lower() or "conditions" in context.lower():
                            checkbox_label = context.strip()
                            break
                    
                    # Also check current line for patterns like "I agree to terms & conditions"
                    if "agree" in line_lower or "terms" in line_lower:
                        # Look for the text after checkbox
                        parts = line.split('checkbox', 1)
                        if len(parts) > 1:
                            remaining = parts[1]
                            # Extract meaningful text
                            text_match = re.search(r'"([^\"]*)"', remaining)
                            if text_match:
                                checkbox_label = text_match.group(1)
                
                result["checkboxes"].append(checkbox_label)
                result["interactives"].append(f"checkbox: {checkbox_label}")
                result["all_clickables"].append(checkbox_label)
                result["refs"][checkbox_label] = refs

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
            
            # If combobox and options exist after this line, capture options
            if detected_type == "combobox" and refs:
                # scan next lines for options
                opts = []
                for j in range(i+1, min(len(lines), i+12)):
                    next_line = _to_str(lines[j]).strip()
                    if 'option' in next_line.lower() and quoted_pattern.search(next_line):
                        q = quoted_pattern.search(next_line).group(1).strip()
                        q = re.sub(r'\s*\[.*\]$', '', q).strip()
                        if q and q not in opts:
                            opts.append(q)
                    else:
                        # stop if no more option lines
                        if next_line and not next_line.lower().startswith('- option'):
                            break
                if opts:
                    # Store each combobox separately with its ref and options
                    for ref in refs:
                        combobox_id = f"combobox_{ref}"
                        if combobox_id not in result["comboboxes"]:
                            result["comboboxes"].append(combobox_id)
                            result["interactives"].append(f"dropdown: {combobox_id}")
                            result["refs"][combobox_id] = [ref]
                            result["refs"][f"{combobox_id}_options"] = opts
            
            # Enhanced label extraction for elements without quoted labels
            if not labels and refs:
                ref = refs[0]
                context_lines = context_map.get(ref, [])
                
                potential_labels = []
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
                
                for context in context_lines:
                    clean_context = re.sub(r'^(Section:|text:\s*)', '', context).strip()
                    clean_context = re.sub(r'[*\"]+', '', clean_context).strip()
                    
                    if (len(clean_context) > 1 and 
                        clean_context.lower() not in ['generic', 'text', 'img', 'please select'] and
                        not clean_context.startswith('- ')):
                        potential_labels.append(clean_context)
                
                if not potential_labels:
                    text_parts = line.split()
                    for part in text_parts:
                        if (len(part) > 2 and 
                            part not in ['generic', 'ref', 'cursor'] and
                            not part.startswith('[') and 
                            not part.endswith(']')):
                            potential_labels.append(part)
                
                if potential_labels:
                    best_label = potential_labels[0]
                    for label in potential_labels:
                        label_lower = label.lower()
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
                    
                    element_info = {
                        "label": label,
                        "type": detected_type,
                        "ref": refs[0] if refs else None,
                        "line": line_stripped
                    }
                    
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
                    elif detected_type == "checkbox" and label not in result["checkboxes"]:
                        result["checkboxes"].append(label)
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
                    
                    interactive_desc = f"{detected_type}: {label}"
                    if interactive_desc not in result["interactives"]:
                        result["interactives"].append(interactive_desc)
                    
                    if detected_type in ["button", "tab", "link"] and label not in result["all_clickables"]:
                        result["all_clickables"].append(label)
                    
                    if refs:
                        for ref in refs:
                            if label not in result["refs"]:
                                result["refs"][label] = []
                            if ref not in result["refs"][label]:
                                result["refs"][label].append(ref)
    
    # Additional processing for iframe content and nested elements
    iframe_count = 0
    for raw_line in lines:
        line = _to_str(raw_line)
        if "iframe" in line.lower():
            iframe_count += 1
            result["form_sections"].append(f"iframe_section_{iframe_count}")
    
    # Extract dropdown/combobox options and radiogroup questions (already enriched above)
    for i, raw_line in enumerate(lines):
        line = _to_str(raw_line)
        line_lower = line.lower()
        
        if "radiogroup" in line_lower:
            labels = quoted_pattern.findall(line)
            refs = ref_pattern.findall(line)
            if labels and refs:
                skill_name = labels[0]
                if skill_name not in result["radio_groups"]:
                    result["radio_groups"].append(skill_name)
                    result["interactives"].append(f"skill_rating: {skill_name}")
                    result["refs"][skill_name] = refs
        
        elif "group" in line_lower:
            labels = quoted_pattern.findall(line)
            refs = ref_pattern.findall(line)
            if labels and refs:
                question = labels[0]
                if question not in result["radio_groups"]:
                    result["radio_groups"].append(question)
                    result["interactives"].append(f"yes_no_question: {question}")
                    result["refs"][question] = refs
        
        elif "combobox" in line_lower and quoted_pattern.search(line):
            labels = quoted_pattern.findall(line)
            refs = ref_pattern.findall(line)
            if labels and refs:
                question = labels[0]
                if question not in result["comboboxes"]:
                    result["comboboxes"].append(question)
                    result["interactives"].append(f"dropdown: {question}")
                    result["refs"][question] = refs
                    # lookahead for options appended earlier in loop; already added to refs as <question>_options if found
    
        elif "radio" in line_lower and not "radiogroup" in line_lower:
            labels = quoted_pattern.findall(line)
            refs = ref_pattern.findall(line)
            if labels and refs:
                option = labels[0]
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
    for i, raw_line in enumerate(lines):
        line_stripped = _to_str(raw_line).strip()
        if (line_stripped.endswith('?') and len(line_stripped) > 10 and not any(kw in line_stripped.lower() for kw in INTERACTIVE_KEYWORDS)):
            has_following_interactive = False
            for j in range(i+1, min(len(lines), i+5)):
                if _is_interactive_line(_to_str(lines[j]).lower()):
                    has_following_interactive = True
                    break
            
            if has_following_interactive:
                clean_question = re.sub(r'^- ', '', line_stripped)
                clean_question = re.sub(r'generic.*?:', '', clean_question).strip()
                if clean_question and clean_question not in question_candidates:
                    question_candidates.append(clean_question)
    
    for question in question_candidates:
        if not any(question in existing for existing in result["comboboxes"] + result["radio_groups"] + result["inputs"]):
            result["interactives"].append(f"question: {question}")
    
    return result

def find_element_ref(snapshot_text: Any, element_text: str, element_type: str = None) -> Optional[str]:
    """
    Enhanced element finding with better pattern matching and iframe support.
    """
    lines = _normalize_snapshot_lines(snapshot_text)
    element_lower = _to_str(element_text).lower()
    
    candidates = []
    
    # Enhanced reference pattern to handle iframe refs
    ref_pattern = re.compile(r"\[ref=((?:f\d+)?e\d+)\]")
    
    for idx, raw_line in enumerate(lines):
        line = _to_str(raw_line)
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
                    nearby_ref_match = ref_pattern.search(_to_str(lines[nearby_idx]))
                    if nearby_ref_match:
                        nearby_ref = nearby_ref_match.group(1)
                        candidates.append((score - 10, nearby_ref, _to_str(lines[nearby_idx]), nearby_idx))
    
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

def analyze_goal(goal_text: str) -> Dict[str, Any]:
    """Enhanced goal analysis with more comprehensive detection."""
    goal_lower = _to_str(goal_text).lower()
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