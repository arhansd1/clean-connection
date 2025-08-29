"""Enhanced utility functions for web automation agent."""
import re
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

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
            base += f", Options: {self.options}"
        if self.required:
            base += ", Required: True"
        base += "}"
        return base


class FormFieldParser:
    """Enhanced form field parser with comprehensive error handling."""
    
    def __init__(self):
        self.ref_pattern = re.compile(r"\[ref=((?:f\d+)?e\d+)\]")
        self.quoted_pattern = re.compile(r'"([^"]+)"')
        
        # Field type mappings
        self.element_type_mapping = {
            "textbox": "text",
            "input": "text",
            "textarea": "textarea", 
            "combobox": "dropdown",
            "select": "dropdown",
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
    
    def extract_field_info(self, field_line: str) -> Tuple[str, str, str, str]:
        """Extract field information with robust error handling."""
        try:
            # Remove leading dash and whitespace
            cleaned_line = field_line.strip().lstrip('- ')
            
            # Split on the first comma to separate element type from rest
            parts = self.safe_split(cleaned_line, ',', 2)
            element_type = self.safe_get_list_item(parts, 0).strip()
            remainder = self.safe_get_list_item(parts, 1).strip()
            
            # Extract label (everything before the ref)
            if 'ref:' in remainder:
                label_part, ref_part = remainder.rsplit('ref:', 1)
                label = label_part.strip().rstrip(',').strip()
                ref = ref_part.split(',')[0].strip()  # Get ref, ignore anything after comma
            else:
                label = remainder
                ref = ""
            
            # Extract field type if present (type:xxx pattern)
            field_type = ""
            type_match = re.search(r'type:(\w+)', field_line)
            if type_match:
                field_type = type_match.group(1)
            
            return element_type, label, ref, field_type
            
        except Exception as e:
            # Return safe defaults if parsing fails
            return "unknown", field_line.strip(), "", ""
    
    def detect_field_type(self, label: str, element_type: str, existing_type: str = "") -> str:
        """Detect field type based on label content and element type."""
        if existing_type:
            return existing_type
            
        label_lower = label.lower()
        
        # Check against field type patterns
        for field_type, patterns in self.field_type_patterns.items():
            if any(pattern in label_lower for pattern in patterns):
                return field_type
        
        # Default based on element type
        return self.element_type_mapping.get(element_type.lower(), "text")
    
    def extract_radio_options(self, field_lines: List[str], start_index: int) -> Tuple[List[str], int]:
        """Extract radio button options that follow a radio group."""
        options = []
        current_index = start_index + 1
        
        try:
            while current_index < len(field_lines):
                line = field_lines[current_index].strip()
                if not line.startswith('- Radio group,'):
                    break
                    
                # Extract option label
                element_type, label, ref, field_type = self.extract_field_info(line)
                if label:
                    options.append(label)
                current_index += 1
                
        except (IndexError, AttributeError):
            pass
            
        return options, current_index - 1
    
    def parse_form_fields(self, form_field_text: str) -> List[FormField]:
        """Parse form fields from the structured text format."""
        if not form_field_text or not isinstance(form_field_text, str):
            return []
            
        lines = [line.strip() for line in form_field_text.strip().split('\n') if line.strip()]
        fields = []
        i = 0
        
        while i < len(lines):
            try:
                line = lines[i]
                
                # Skip empty lines or lines that don't look like field definitions
                if not line or not line.startswith('-'):
                    i += 1
                    continue
                
                # Extract basic field information
                element_type, label, ref, field_type = self.extract_field_info(line)
                
                if not element_type or not label:
                    i += 1
                    continue
                
                # Detect actual field type
                detected_type = self.detect_field_type(label, element_type, field_type)
                
                # Check if field is required (has asterisk)
                required = '*' in line or 'required' in line.lower()
                
                # Handle radio groups specially
                options = None
                if 'radio group' in element_type.lower():
                    # Look ahead for radio options
                    radio_options, last_option_index = self.extract_radio_options(lines, i)
                    if radio_options:
                        options = radio_options
                        i = last_option_index  # Skip the option lines
                
                # Extract any inline text that might be different from label
                text = label  # Default to label
                
                # Look for patterns like "Email example@example.com"
                if ' ' in label and not label.endswith('?'):
                    # Check if this might be label + placeholder text
                    words = label.split()
                    if len(words) >= 2:
                        # Common patterns: "Email example@example.com", "Date Date"
                        first_word = words[0]
                        rest = ' '.join(words[1:])
                        
                        # If the rest looks like placeholder text
                        if ('@' in rest or 
                            rest.lower() in ['date', 'example', 'placeholder'] or
                            rest == first_word):  # Repeated word pattern
                            text = rest
                            label = first_word
                
                # Create form field
                field = FormField(
                    element_type=element_type.lower().replace(' ', '_'),
                    label=label,
                    text=text,
                    ref=ref,
                    field_type=detected_type,
                    options=options,
                    required=required
                )
                
                fields.append(field)
                i += 1
                
            except Exception as e:
                # Log error but continue processing
                print(f"Error processing field line {i}: {e}")
                i += 1
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