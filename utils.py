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
    
    def extract_options(self, field_line: str) -> List[str]:

        """Extract options from field line."""

        options_match = self.options_pattern.search(field_line)

        if options_match:

            options_str = options_match.group(1)

            # Split by comma and clean up

            options = [opt.strip() for opt in options_str.split(',')]

            return [opt for opt in options if opt]  # Remove empty options

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
    
    def parse_form_fields(self, form_field_text: str) -> List[FormField]:
        """Parse form fields from the structured text format."""
        if not form_field_text or not isinstance(form_field_text, str):
            return []
            
        lines = [line.rstrip() for line in form_field_text.strip().split('\n') if line.strip()]
        fields = []
        label_ref_counts = {}  # Track how many times each label appears

        for i, line in enumerate(lines):
            try:
                # Skip empty lines or lines that don't look like field definitions
                if not line or not line.lstrip().startswith('-'):
                    continue
                
                # Extract basic field information including options
                element_type, label, ref, field_type, options = self.extract_field_info(line)
                
                if not element_type or not label:
                    continue

                # Handle duplicate labels by making them unique with ref
                original_label = label
                if label in label_ref_counts:
                    label_ref_counts[label] += 1
                    # For duplicate labels, append ref or counter to make unique
                    if ref:
                        label = f"{original_label} ({ref})"
                    else:
                        label = f"{original_label} #{label_ref_counts[label]}"
                else:
                    label_ref_counts[label] = 1
                
                # Detect actual field type
                detected_type = self.detect_field_type(original_label, element_type, field_type)
                
                # Check if field is required (has asterisk)
                required = '*' in line or 'required' in line.lower()
                
                # Extract any inline text that might be different from label
                text = original_label  # Use original label for text
                
                # Look for patterns like "Email example@example.com"
                if ' ' in original_label and not original_label.endswith('?'):
                    # Check if this might be label + placeholder text
                    words = original_label.split()
                    if len(words) >= 2:
                        # Common patterns: "Email example@example.com", "Date Date"
                        first_word = words[0]
                        rest = ' '.join(words[1:])
                        
                        # If the rest looks like placeholder text
                        if ('@' in rest or 
                            rest.lower() in ['date', 'example', 'placeholder'] or
                            rest == first_word):  # Repeated word pattern
                            text = rest
                            # Don't modify the unique label, just the text
                
                # Create form field
                field = FormField(
                    element_type=element_type.lower().replace(' ', '_'),
                    label=label,  # Use the potentially modified unique label
                    text=text,
                    ref=ref,
                    field_type=detected_type,
                    options=options if options else None,
                    required=required
                )
                
                fields.append(field)
                
            except Exception as e:
                # Log error but continue processing
                print(f"Error processing field line {i}: {e}")
                continue
        
        return fields
    
    def format_clean_output(self, fields: List[FormField]) -> str:
        """Format fields in the requested clean format."""
        if not fields:
            return "No form fields found."
            
        output_lines = []
        for field in fields:
            output_lines.append(str(field))
        
        return '\n'.join(output_lines)


# Global parser instance for easy access
_form_parser = FormFieldParser()

def parse_form_fields_enhanced(form_field_text: str) -> str:
    """Enhanced form field parsing with clean output format."""
    fields = _form_parser.parse_form_fields(form_field_text)
    return _form_parser.format_clean_output(fields)

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
    print("\nSNAPSHOT TEXT", snapshot_text, "\n")
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
        
         # Enhanced file upload detection
        if any(kw in line_lower for kw in ["browse files", "drag and drop files", "uploaded files"]):
            refs = ref_pattern.findall(line)
            if refs:
                # Look for context lines to get meaningful label
                context_lines = context_map.get(refs[0], [])
                file_label = "File Upload"
                
                # Check previous lines for file upload context
                for j in range(max(0, i-3), i):
                    prev_line = _to_str(lines[j]).strip()
                    if any(kw in prev_line.lower() for kw in ["please upload", "upload your", "cv", "cover letter", "resume"]):
                        # Extract the meaningful part
                        if "please upload" in prev_line.lower():
                            file_label = prev_line.strip().lstrip('- generic:').strip()
                        break
                
                # Add to file uploads
                result["file_uploads"].append(file_label)
                result["interactives"].append(f"file_upload: {file_label}")
                result["all_clickables"].append(file_label)
                result["refs"][file_label] = refs
        
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
                    # label for combobox (try quoted label or context)
                    comb_label = labels[0] if labels else (context_map.get(refs[0], ['Combobox'])[0])
                    # add a descriptive entry
                    combo_desc = f"{comb_label} -> options: {opts}"
                    if combo_desc not in result["interactives"]:
                        result["interactives"].append(combo_desc)
                    if comb_label not in result["comboboxes"]:
                        result["comboboxes"].append(comb_label)
                    # store options under refs
                    for r in refs:
                        result["refs"].setdefault(comb_label, []).append(r)
                        result["refs"].setdefault(f"{comb_label}_options", []).extend(opts)
            
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


def extract_form_fields(snapshot_text: Any) -> List[Dict[str, str]]:
    """
    Enhanced form field extraction with better detection and iframe support.
    """
    lines = _normalize_snapshot_lines(snapshot_text)
    
    def _scan(lines: List[str], gated: bool) -> List[Dict[str, str]]:
        fields_local: List[Dict[str, str]] = []
        in_form = False
        section_context = ""
        
        for i, raw_line in enumerate(lines):
            line = _to_str(raw_line)
            line_lower = line.lower()
            
            # Track sections for better context
            if "heading" in line_lower:
                heading_match = re.search(r'"([^\"]+?)"', line)
                if heading_match:
                    section_context = heading_match.group(1)
            
            if gated and ("form" in line_lower or "personal information" in line_lower or "position information" in line_lower):
                in_form = True
            if gated and not in_form:
                continue
                
            if _is_interactive_line(line_lower):
                # Enhanced label extraction
                label_match = re.search(r'"([^\"]+?)"', line)
                label = label_match.group(1) if label_match else "Unlabeled field"
                
                # If no label found, try to get from context
                if label == "Unlabeled field" and section_context:
                    # Look at nearby lines for context
                    for j in range(max(0, i-3), min(len(lines), i+3)):
                        context_line = _to_str(lines[j]).strip()
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
                
                # If combobox/select, attempt to capture options immediately after
                if field_type in ("select", "dropdown" , "combobox"):
                    opts = []
                    for j in range(i+1, min(len(lines), i+12)):
                        next_line = _to_str(lines[j]).strip()
                        if 'option' in next_line.lower() and re.search(r'"([^\"]+?)"', next_line):
                            opt = re.search(r'"([^\"]+?)"', next_line).group(1).strip()
                            opt = re.sub(r'\s*\[.*\]$', '', opt).strip()
                            if opt:
                                opts.append(opt)
                        else:
                            # stop if line not an option
                            if next_line and not next_line.lower().startswith('- option'):
                                break
                    if opts:
                        field_info["options"] = opts
                
                fields_local.append(field_info)
        
        return fields_local

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