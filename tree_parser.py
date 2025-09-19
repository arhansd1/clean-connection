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

def parse_snapshot(text):
    """Parse accessibility snapshot text and extract structured form fields."""
    lines = text.splitlines()
    start = 0
    for i,l in enumerate(lines):
        if l.strip() == 'yaml':
            start = i+1
            break
    else:
        for i,l in enumerate(lines):
            if l.lstrip().startswith('- '):
                start = i
                break
    snapshot_lines = lines[start:]
    tree = build_tree(snapshot_lines)
    return extract_fields(tree)