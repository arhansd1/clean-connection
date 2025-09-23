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

import re, json
from collections import OrderedDict

def _extract_bracket_tokens(s):
    return re.findall(r'\[([^\]]+)\]', s)

def _parse_attrs(tokens):
    attrs = {}
    for t in tokens:
        if '=' in t:
            k, v = t.split('=', 1)
            attrs[k.strip()] = v.strip()
        else:
            attrs[t.strip()] = True
    return attrs

def _strip_brackets(s):
    return re.sub(r'\s*\[[^\]]*\]', '', s).strip()

def parse_line_to_node(raw_content):
    # Trim outer single quotes that sometimes wrap a whole line in your snapshots
    content_raw = raw_content
    if content_raw.startswith("'") and content_raw.endswith("'"):
        content_raw = content_raw[1:-1].strip()
    tokens = _extract_bracket_tokens(content_raw)
    attrs = _parse_attrs(tokens)
    node = {"raw": raw_content, "attrs": attrs, "children": []}
    node['ref'] = attrs.get('ref')

    # remove bracket parts to simplify content parsing
    content = _strip_brackets(content_raw)

    # match: type optionally followed by "quoted label" or 'quoted label' or : after-label
    m = re.match(r'^(?P<type>\w+)(?:\s+(?:["\'](?P<quoted>[^"\']+)["\']))?(?:\s*:\s*(?P<after>.*))?$', content)
    if m:
        t = m.group('type')
        node['type'] = t
        quoted = m.group('quoted')
        after = m.group('after')
        if quoted:
            node['label'] = quoted
        elif after is not None and after != "":
            # remove surrounding quotes if present
            label = after.strip()
            if (label.startswith('"') and label.endswith('"')) or (label.startswith("'") and label.endswith("'")):
                label = label[1:-1]
            node['label'] = label
        else:
            node['label'] = None
    else:
        parts = content.split(None,1)
        node['type'] = parts[0] if parts else 'generic'
        node['label'] = parts[1] if len(parts)>1 else None

    node['has_children'] = raw_content.strip().endswith(':')
    return node

def build_tree(lines):
    root = {"type":"root","children":[],"indent":-1, "parent": None}
    stack = [root]
    for line in lines:
        if not line.strip():
            continue
        lstrip = line.lstrip()
        if not lstrip.startswith('-'):
            continue
        indent = len(line) - len(lstrip)
        content = lstrip[1:].lstrip()
        node = parse_line_to_node(content)
        node['indent'] = indent
        node['parent'] = None
        while stack and stack[-1]['indent'] >= indent:
            stack.pop()
        parent = stack[-1]
        parent.setdefault('children', []).append(node)
        node['parent'] = parent
        stack.append(node)
    return root

def extract_fields(tree):
    result = OrderedDict()

    def add_field(key, field):
        if key not in result:
            result[key] = []
        result[key].append(field)

    def node_has_input_child(node):
        for ch in node.get('children', []):
            if ch.get('type') in ('textbox','combobox','radio','checkbox','button','spinbutton','file','file_upload','radiogroup'):
                return True
            if ch.get('children'):
                if node_has_input_child(ch):
                    return True
        return False

    def find_nearby_label(node):
        # Search ancestors for nearby label siblings.
        anc = node.get('parent')
        path = []
        while anc:
            path.append(anc)
            anc = anc.get('parent')
        # path: immediate parent first, then grandparent...
        for ancestor in path:
            children = ancestor.get('children', [])
            # find the top-level child in this ancestor that contains our node
            top_child = node
            while top_child.get('parent') and top_child.get('parent') is not ancestor:
                top_child = top_child.get('parent')
            # now find index
            try:
                idx = children.index(top_child)
            except ValueError:
                idx = None
            # search left neighbors first for a label node
            if idx is not None:
                # look left
                for i in range(idx-1, -1, -1):
                    sib = children[i]
                    if sib.get('label') and sib.get('type') in ('text','generic','paragraph','heading'):
                        return sib.get('label')
                    # if sib itself contains a text child label, prefer that
                    # look inside sib for first text/generic with label
                    label_in = find_label_in_subtree(sib)
                    if label_in:
                        return label_in
                # then look right
                for i in range(idx+1, len(children)):
                    sib = children[i]
                    if sib.get('label') and sib.get('type') in ('text','generic','paragraph','heading'):
                        return sib.get('label')
                    label_in = find_label_in_subtree(sib)
                    if label_in:
                        return label_in
        return None

    def find_label_in_subtree(node):
        # DFS to find first text/generic/paragraph/heading with a label
        if node.get('label') and node.get('type') in ('text','generic','paragraph','heading'):
            return node.get('label')
        for ch in node.get('children', []):
            lbl = find_label_in_subtree(ch)
            if lbl:
                return lbl
        return None

    def process_node(node, group_label=None):
        t = node.get('type')
        label = node.get('label')
        attrs = node.get('attrs', {})
        children = node.get('children', [])

        # If node is a generic/text and has visible label and subtree contains inputs -> grouping label
        if t in ('generic','text','paragraph') and label and node_has_input_child(node):
            group_label = label

        if t == 'textbox':
            placeholder = None
            for ch in children:
                if ch.get('type') in ('generic','text') and ch.get('label'):
                    txt = ch['label'].strip()
                    if '@' in txt or len(txt.split()) <= 4:
                        placeholder = txt
                        break
            field_label = label or group_label or find_nearby_label(node) or 'textbox'
            field = {"ref": node.get('ref'), "type": "textbox", "placeholder": placeholder}
            add_field(field_label, field)

        elif t == 'combobox':
            options = []
            selected_val = None
            for ch in children:
                if ch.get('type') == 'option':
                    opt = ch.get('label') or ''
                    options.append(opt)
                    if ch.get('attrs',{}).get('selected', False):
                        selected_val = opt
            field_label = label or group_label or find_nearby_label(node) or 'combobox'
            field = {"ref": node.get('ref'), "type": "combobox", "options": options}
            if selected_val is not None:
                field['selected'] = selected_val
            add_field(field_label, field)

        elif t == 'radio':
            field_label = group_label or label or find_nearby_label(node) or 'radio'
            checked = bool(attrs.get('checked', False))
            field = {"ref": node.get('ref'), "type": "radio", "label": label, "checked": checked}
            add_field(field_label, field)

        elif t == 'checkbox':
            field_label = label or group_label or find_nearby_label(node) or 'checkbox'
            checked = bool(attrs.get('checked', False))
            field = {"ref": node.get('ref'), "type": "checkbox", "checked": checked}
            add_field(field_label, field)

        elif t == 'spinbutton':
            field_label = label or group_label or find_nearby_label(node) or 'spinbutton'
            field = {"ref": node.get('ref'), "type": "spinbutton", "label": label}
            add_field(field_label, field)

        elif t == 'button':
            field_label = label or group_label or find_nearby_label(node) or 'button'
            field = {"ref": node.get('ref'), "type": "button", "label": label}
            add_field(field_label, field)

        elif t in ('iframe','main','root','region','tabpanel'):
            for ch in children:
                process_node(ch, group_label=group_label)

        elif t in ('heading','img','link','paragraph','text','generic','separator','tab','tablist','list'):
            if t in ('paragraph','heading','text') and label and node_has_input_child(node):
                group_label = label
            for ch in children:
                process_node(ch, group_label=group_label if group_label else label)

        elif t in ('radiogroup',):
            for ch in children:
                process_node(ch, group_label=label or group_label)

        else:
            for ch in children:
                process_node(ch, group_label=group_label if group_label else label)

    for child in tree.get('children', []):
        process_node(child, group_label=None)
    return result


# BUCKETING 
def extract_interactive_elements(snapshot_text):
    lines = _normalize_snapshot_lines(snapshot_text)

    start = 0
    for i, l in enumerate(lines):
        if l.strip() == 'yaml':
            start = i + 1
            break
    else:
        for i, l in enumerate(lines):
            if l.lstrip().startswith('- '):
                start = i
                break

    snapshot_lines = lines[start:]
    tree = build_tree(snapshot_lines)
    fields = extract_fields(tree)

    # 🔹 Debug
    #print("Extracted fields:", json.dumps(fields, indent=2))

    refs = {}
    inputs, checkboxes, radio_groups, comboboxes, file_uploads, dropdowns, buttons = (
        [], [], [], [], [], [], []
    )

    for label, items in fields.items():
        for item in items:
            ref = item.get("ref")
            ftype = item.get("type", "").lower()

            if not ref:
                continue

            refs.setdefault(label, []).append(ref)

            if ftype in ("textbox", "input"):
                inputs.append(label)
            elif ftype in ("checkbox",):
                checkboxes.append(label)
            elif ftype in ("radio", "radiogroup"):
                radio_groups.append(label)
            elif ftype in ("combobox", "select", "dropdown"):
                comboboxes.append(f"combobox_{ref}")
                if "options" in item:
                    refs[f"combobox_{ref}_options"] = item["options"]
            elif ftype in ("file", "file_upload"):
                file_uploads.append(f"file_upload_{ref}")
            elif ftype in ("button",):
                buttons.append(label)

    return {
        "raw_fields": fields,
        "refs": refs,
        "inputs": inputs,
        "checkboxes": checkboxes,
        "radio_groups": radio_groups,
        "comboboxes": comboboxes,
        "file_uploads": file_uploads,
        "buttons": buttons,
    }




# def parse_snapshot(text):
#     lines = text.splitlines()
#     start = 0
#     for i,l in enumerate(lines):
#         if l.strip() == 'yaml':
#             start = i+1
#             break
#     else:
#         for i,l in enumerate(lines):
#             if l.lstrip().startswith('- '):
#                 start = i
#                 break
#     snapshot_lines = lines[start:]
#     tree = build_tree(snapshot_lines)
#     return extract_fields(tree)


# import re
# from typing import Any, Dict, List, Optional

# _REF_RE = re.compile(r"\[ref=((?:f\d+)?e\d+)\]")
# _LABEL_RE = re.compile(r'"([^"]+)"|\'([^\']+)\'', re.UNICODE)

# # Keep these in sync with your other utilities
# _INPUT_TOKENS = (
#     "textbox", "combobox", "textarea", "select", "checkbox",
#     "radio", "radiogroup", "spinbutton", "file_upload", "file", "input"
# )

# def _dedup(seq: List[str]) -> List[str]:
#     seen: set = set()
#     out: List[str] = []
#     for x in seq:
#         if x not in seen:
#             seen.add(x)
#             out.append(x)
#     return out

# def extract_interactive_elements(snapshot_text: Any) -> Dict[str, Any]:
#     """
#     Return a structure that agent_core.py expects:

#     {
#       "title": str|None,
#       "buttons": [str],
#       "tabs": [str],
#       "inputs": [str],
#       "links": [str],
#       "refs": { label: [ref, ...] }
#     }
#     """
#     lines: List[str] = _normalize_snapshot_lines(snapshot_text)

#     title: Optional[str] = None
#     buttons: List[str] = []
#     tabs: List[str] = []
#     inputs: List[str] = []
#     links: List[str] = []
#     refs: Dict[str, List[str]] = {}

#     # Try to find a page title quickly
#     for ln in lines[:80]:  # only early lines
#         low = ln.lower()
#         # common patterns seen in snapshots
#         m = re.search(r'^\s*title\s*[:=]\s*(.+)$', ln, flags=re.I)
#         if m:
#             title = m.group(1).strip().strip('"\'')
#             break
#         if "heading" in low or "page title" in low:
#             ml = _LABEL_RE.search(ln)
#             if ml:
#                 title = (ml.group(1) or ml.group(2) or "").strip()
#                 if title:
#                     break

#     def _label_from_line(line: str) -> Optional[str]:
#         m = _LABEL_RE.search(line)
#         if not m:
#             return None
#         return (m.group(1) or m.group(2) or "").strip()

#     for line in lines:
#         low = line.lower()
#         ref_m = _REF_RE.search(line)
#         ref = ref_m.group(1) if ref_m else None
#         label = _label_from_line(line)

#         kind: Optional[str] = None
#         if "button" in low:
#             kind = "buttons"
#         elif "tab" in low and "table" not in low:  # avoid table rows
#             kind = "tabs"
#         elif "link" in low or "anchor" in low:
#             kind = "links"
#         elif any(tok in low for tok in _INPUT_TOKENS):
#             kind = "inputs"

#         if kind:
#             text = label or (ref or line.strip())
#             if kind == "buttons":
#                 buttons.append(text)
#             elif kind == "tabs":
#                 tabs.append(text)
#             elif kind == "links":
#                 links.append(text)
#             else:
#                 inputs.append(text)

#             if ref and text:
#                 refs.setdefault(text, []).append(ref)

#     return {
#         "title": title,
#         "buttons": _dedup(buttons),
#         "tabs": _dedup(tabs),
#         "inputs": _dedup(inputs),
#         "links": _dedup(links),
#         "refs": refs
#     }




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


# def extract_form_fields(snapshot_text: Any) -> List[Dict[str, str]]:
#     """
#     Enhanced form field extraction with better detection and iframe support.
#     """
#     lines = _normalize_snapshot_lines(snapshot_text)
    
#     def _scan(lines: List[str], gated: bool) -> List[Dict[str, str]]:
#         fields_local: List[Dict[str, str]] = []
#         in_form = False
#         section_context = ""
        
#         for i, raw_line in enumerate(lines):
#             line = _to_str(raw_line)
#             line_lower = line.lower()
            
#             # Track sections for better context
#             if "heading" in line_lower:
#                 heading_match = re.search(r'"([^\"]+?)"', line)
#                 if heading_match:
#                     section_context = heading_match.group(1)
            
#             if gated and ("form" in line_lower or "personal information" in line_lower or "position information" in line_lower):
#                 in_form = True
#             if gated and not in_form:
#                 continue
                
#             if _is_interactive_line(line_lower):
#                 # Enhanced label extraction
#                 label_match = re.search(r'"([^\"]+?)"', line)
#                 label = label_match.group(1) if label_match else "Unlabeled field"
                
#                 # If no label found, try to get from context
#                 if label == "Unlabeled field" and section_context:
#                     # Look at nearby lines for context
#                     for j in range(max(0, i-3), min(len(lines), i+3)):
#                         context_line = _to_str(lines[j]).strip()
#                         if (context_line and 
#                             not re.search(r'\[ref=', context_line) and
#                             not any(kw in context_line.lower() for kw in INTERACTIVE_KEYWORDS)):
#                             # Clean up context
#                             clean_context = re.sub(r'^- ', '', context_line)
#                             clean_context = re.sub(r':$', '', clean_context)
#                             if len(clean_context) > 1 and clean_context != "generic":
#                                 label = clean_context
#                                 break
                
#                 # Determine field type with enhanced detection
#                 field_type = "text"
#                 type_mappings = {
#                     "textbox": "text",
#                     "input": "text", 
#                     "textarea": "textarea",
#                     "combobox": "select",
#                     "select": "select",
#                     "checkbox": "checkbox",
#                     "radio": "radio",
#                     "radiogroup": "radio",
#                     "spinbutton": "number",
#                     "button": "button"
#                 }
                
#                 for keyword, mapped_type in type_mappings.items():
#                     if keyword in line_lower:
#                         field_type = mapped_type
#                         break
                
#                 # Special field type detection
#                 if any(kw in label.lower() for kw in ["email", "@"]):
#                     field_type = "email"
#                 elif any(kw in label.lower() for kw in ["phone", "tel"]):
#                     field_type = "tel"
#                 elif any(kw in label.lower() for kw in ["date", "calendar"]):
#                     field_type = "date"
#                 elif any(kw in label.lower() for kw in ["upload", "file", "cv", "resume"]):
#                     field_type = "file"
#                 elif any(kw in label.lower() for kw in ["password", "pass"]):
#                     field_type = "password"
                
#                 required = "required" in line_lower or "*" in line
                
#                 field_info = {
#                     "label": label,
#                     "type": field_type,
#                     "required": required,
#                     "section": section_context
#                 }
                
#                 # If combobox/select, attempt to capture options immediately after
#                 if field_type in ("select", "dropdown" , "combobox"):
#                     opts = []
#                     for j in range(i+1, min(len(lines), i+12)):
#                         next_line = _to_str(lines[j]).strip()
#                         if 'option' in next_line.lower() and re.search(r'"([^\"]+?)"', next_line):
#                             opt = re.search(r'"([^\"]+?)"', next_line).group(1).strip()
#                             opt = re.sub(r'\s*\[.*\]$', '', opt).strip()
#                             if opt:
#                                 opts.append(opt)
#                         else:
#                             # stop if line not an option
#                             if next_line and not next_line.lower().startswith('- option'):
#                                 break
#                     if opts:
#                         field_info["options"] = opts
                
#                 fields_local.append(field_info)
        
#         return fields_local

#     fields = _scan(lines, gated=True)
#     if not fields:
#         fields = _scan(lines, gated=False)
    
#     # Enhanced deduplication - keep most informative version
#     dedup = {}
#     for f in fields:
#         label = f["label"]
#         if label not in dedup or (f.get("section") and not dedup[label].get("section")):
#             dedup[label] = f
    
#     return list(dedup.values())




#used in main.py
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
