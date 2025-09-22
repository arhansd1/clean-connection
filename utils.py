"""Simplified utilities for form field extraction and data matching."""
import re
from typing import Dict, List, Any, Optional

def extract_form_fields(snapshot: str) -> List[Dict[str, str]]:
    """Extract form fields from snapshot with their refs, optimized for JotForm."""
    fields = []
    if not snapshot:
        return fields
        
    lines = snapshot.split('\n')
    current_section = None
    
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
            
        # Extract ref from line - JotForm uses e.g., [ref=e123]
        ref_match = re.search(r'\[ref=(e\d+)\]', line)
        if not ref_match:
            # Check if this is a section header
            section_match = re.search(r'^(\d+\.\s*)?([^\[\]]+?)(?:\s*\[.*\])?$', line.strip('"'))
            if section_match:
                current_section = section_match.group(2).strip()
            continue
            
        ref = ref_match.group(1)
        
        # Look for label in current or previous lines
        label = ""
        
        # Check current line for label in quotes or after ref
        label_match = re.search(r'"([^"]+)"', line)
        if label_match:
            label = label_match.group(1)
        else:
            # Check previous lines for potential label
            for j in range(max(0, i-3), i):
                prev_line = lines[j].strip()
                if not prev_line or 'ref=' in prev_line:
                    continue
                # Simple heuristic: take the first non-empty line before the input
                label = prev_line.strip('"')
                break
        
        # Clean up label
        if label:
            label = re.sub(r'\s+', ' ', label).strip()
            label = re.sub(r'[^\w\s]', '', label)  # Remove special chars
            
        # If no label found, use the ref as a fallback
        if not label:
            label = f"Field {ref}"
            
        # Add section to label if available
        if current_section and not label.startswith(current_section):
            label = f"{current_section}: {label}"
        
        # Determine field type and action
        line_lower = line.lower()
        
        # Check for specific JotForm classes
        if 'textbox' in line_lower or 'input' in line_lower or 'textarea' in line_lower or 'jfField' in line_lower:
            field_type = "text"
            action = "type"
        elif 'combobox' in line_lower or 'select' in line_lower or 'dropdown' in line_lower:
            field_type = "select"
            action = "select"
        elif 'file' in line_lower or 'upload' in line_lower or 'jfUploadButton' in line_lower:
            field_type = "file"
            action = "upload"
        elif 'checkbox' in line_lower or 'radio' in line_lower:
            field_type = "checkbox"
            action = "click"
        elif 'button' in line_lower or 'submit' in line_lower or 'next' in line_lower:
            field_type = "button"
            action = "click"
        else:
            # Default to text input
            field_type = "text"
            action = "type"
        
        fields.append({
            "type": field_type,
            "label": label,
            "ref": ref,
            "action": action,
            "line": line
        })
    
    return fields

def match_user_data_to_fields(fields: List[Dict[str, str]], user_data: Dict[str, str]) -> List[Dict[str, str]]:
    """Match user data to form fields based on field labels and types with improved matching."""
    matched = []
    used_fields = set()
    
    # Define field mappings with improved matching
    field_mappings = {
        'name': {'keywords': ['name', 'full name', 'your name', 'applicant name'], 'type': 'text'},
        'first_name': {'keywords': ['first name', 'firstname', 'fname', 'given name', 'forename'], 'type': 'text'},
        'last_name': {'keywords': ['last name', 'lastname', 'lname', 'surname', 'family name'], 'type': 'text'},
        'email': {'keywords': ['email', 'e-mail', 'email address', 'mail', 'e mail'], 'type': 'email'},
        'phone': {'keywords': ['phone', 'mobile', 'telephone', 'contact number', 'phone number', 'cell', 'mobile number'], 'type': 'tel'},
        'address': {'keywords': ['address', 'street address', 'location', 'mailing address'], 'type': 'text'},
        'city': {'keywords': ['city', 'town', 'city/town'], 'type': 'text'},
        'state': {'keywords': ['state', 'province', 'region', 'state/province'], 'type': 'text'},
        'country': {'keywords': ['country', 'nation', 'country of residence'], 'type': 'text'},
        'zipcode': {'keywords': ['zip', 'zipcode', 'zip code', 'postal code', 'pincode', 'postcode'], 'type': 'text'},
        'experience': {'keywords': ['experience', 'work experience', 'years of experience', 'professional experience'], 'type': 'text'},
        'skills': {'keywords': ['skills', 'skill', 'technical skills', 'expertise', 'skillset'], 'type': 'text'},
        'education': {'keywords': ['education', 'qualification', 'degree', 'academic', 'educational background'], 'type': 'text'},
        'company': {'keywords': ['company', 'current company', 'employer', 'organization', 'current employer'], 'type': 'text'},
        'position': {'keywords': ['position', 'job title', 'role', 'designation', 'current position'], 'type': 'text'},
        'salary': {'keywords': ['salary', 'expected salary', 'current salary', 'compensation', 'expected pay'], 'type': 'number'},
        'linkedin': {'keywords': ['linkedin', 'linkedin profile', 'linkedin url', 'linkedin.com'], 'type': 'url'},
        'portfolio': {'keywords': ['portfolio', 'website', 'personal website', 'portfolio url'], 'type': 'url'},
        'github': {'keywords': ['github', 'github profile', 'git', 'github.com'], 'type': 'url'},
        'resume': {'keywords': ['resume', 'cv', 'curriculum vitae', 'upload resume'], 'type': 'file'},
        'cover_letter': {'keywords': ['cover letter', 'message', 'additional information', 'why should we hire you'], 'type': 'textarea'},
        'availability': {'keywords': ['availability', 'notice period', 'joining date', 'when can you start', 'availability date'], 'type': 'text'},
        'date': {'keywords': ['date', 'today\'s date', 'current date'], 'type': 'date'},
        'signature': {'keywords': ['signature', 'e-signature', 'digital signature'], 'type': 'text'}
    }
    
    # First pass: try to match fields exactly
    for field in fields:
        if field['ref'] in used_fields:
            continue
            
        label_lower = field['label'].lower()
        field_type = field.get('type', 'text')
        best_match = None
        best_score = 0
        
        # Find best matching user data key
        for user_key, mapping in field_mappings.items():
            if user_key not in user_data or user_key in used_fields:
                continue
                
            field_keywords = mapping['keywords']
            expected_type = mapping['type']
            
            # Type compatibility check
            type_score = 0
            if field_type == expected_type or expected_type == 'text':
                type_score = 20
            elif field_type in ['text', 'textarea'] and expected_type in ['text', 'textarea']:
                type_score = 15
            elif field_type in ['email', 'tel', 'url'] and expected_type in ['text', 'email', 'tel', 'url']:
                type_score = 10
                
            if type_score == 0:
                continue  # Skip incompatible types
                
            score = 0
            
            # Exact match gets highest score
            if user_key.replace('_', ' ') == label_lower:
                score = 100
            elif user_key.replace('_', '') == label_lower.replace(' ', ''):
                score = 95
            elif any(keyword == label_lower for keyword in field_keywords):
                score = 90
            else:
                # Check keyword matches
                for keyword in field_keywords:
                    if keyword in label_lower:
                        score = max(score, 80)
                    elif any(word in label_lower for word in keyword.split()):
                        score = max(score, 60)
            
            # Add type score and check if this is the best match so far
            score += type_score
            
            # Additional type-specific scoring
            if field_type == 'file' and user_key in ['resume', 'cv']:
                score += 20
            elif field_type == 'text' and user_key == 'email' and '@' in user_data[user_key]:
                score += 15
            elif field_type == 'tel' and user_key == 'phone' and any(c.isdigit() for c in user_data[user_key]):
                score += 15
            elif field_type == 'url' and user_key in ['linkedin', 'github', 'portfolio']:
                score += 15
                
            if score > best_score and score >= 60:  # Minimum threshold
                best_score = score
                best_match = user_key
        
        # Add field with matched data if found
        if best_match and best_score >= 60:  # Minimum confidence threshold
            # Skip if we've already used this field
            if best_match in used_fields:
                continue
                
            matched.append({
                **field,
                "value": user_data[best_match],
                "user_key": best_match,
                "confidence": best_score
            })
            print(f"Matched '{field['label']}' -> {best_match}: '{user_data[best_match][:50]}...' (confidence: {best_score}%)")
            used_fields.add(best_match)
    
    # Second pass: try to match remaining fields with lower confidence
    for field in fields:
        if field['ref'] in [f['ref'] for f in matched]:
            continue
            
        label_lower = field['label'].lower()
        field_type = field.get('type', 'text')
        
        # Try to find a partial match
        for user_key, value in user_data.items():
            if user_key in used_fields:
                continue
                
            # Check if any word from the label matches the user key
            label_words = set(re.findall(r'\w+', label_lower))
            key_words = set(re.findall(r'\w+', user_key.lower()))
            
            if label_words.intersection(key_words):
                matched.append({
                    **field,
                    "value": value,
                    "user_key": user_key,
                    "confidence": 40  # Low confidence partial match
                })
                print(f"Partially matched '{field['label']}' -> {user_key}: '{value[:50]}...' (low confidence)")
                used_fields.add(user_key)
                break
    
    return matched

def find_element_ref(snapshot_text: str, element_text: str) -> Optional[str]:
    """
    Find element reference in snapshot text.
    
    Args:
        snapshot_text: The page snapshot text
        element_text: Text to search for in the snapshot
        
    Returns:
        str: The element reference (e.g., 'e123') if found, None otherwise
    """
    if not snapshot_text or not element_text:
        return None
        
    lines = snapshot_text.split('\n')
    element_lower = element_text.lower().strip()
    
    # First try exact match
    for line in lines:
        if element_lower == line.lower().strip():
            # Look for ref in surrounding lines
            for i in range(max(0, lines.index(line) - 3), min(len(lines), lines.index(line) + 3)):
                ref_match = re.search(r'\[ref=(e\d+)\]', lines[i])
                if ref_match:
                    return ref_match.group(1)
    
    # Then try partial match
    for line in lines:
        if element_lower in line.lower():
            ref_match = re.search(r'\[ref=(e\d+)\]', line)
            if ref_match:
                return ref_match.group(1)
                
    # Finally, try to find any element that contains the text
    element_words = set(re.findall(r'\w+', element_lower))
    for line in lines:
        line_lower = line.lower()
        line_words = set(re.findall(r'\w+', line_lower))
        
        # If any word matches and we can find a ref
        if element_words.intersection(line_words):
            ref_match = re.search(r'\[ref=(e\d+)\]', line)
            if ref_match:
                return ref_match.group(1)
            ref_match = re.search(r'\[ref=(e\d+)\]', line)
            if ref_match:
                return ref_match.group(1)
    
    return None