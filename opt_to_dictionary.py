import xml.etree.ElementTree as ET
import json
import re

def get_category_for_archetype(archetype_id):
    arch_lower = archetype_id.lower()
    if 'problem_diagnosis' in arch_lower:
        return 'diagnosis'
    elif 'observation' in arch_lower:
        # Some observations are vital signs, some are lab tests
        if any(v in arch_lower for v in ['blood_pressure', 'body_temperature', 'pulse', 'respiration', 'pulse_oximetry', 'body_mass_index', 'height', 'body_weight']):
            return 'vital_sign'
        if 'laboratory' in arch_lower:
            return 'lab_test'
        return 'measurement'
    elif 'symptom' in arch_lower:
        return 'symptom'
    elif 'procedure' in arch_lower:
        return 'procedure'
    return 'condition'

PREFIXES = [
    "acute", "chronic", "severe", "mild", "moderate", "recurrent", "persistent", 
    "intermittent", "bilateral", "unilateral", "left", "right", "upper", "lower", 
    "anterior", "posterior", "suspected", "confirmed", "probable", "possible", 
    "early", "late", "advanced", "progressive", "compensated", "decompensated", 
    "stable", "unstable", "primary", "secondary", "exacerbation of"
]

def generate_expansions(base_term, archetype_id, at_code, category):
    expansions = {}
    # Base term
    expansions[base_term] = {
        "archetype_id": archetype_id,
        "source": "keyword",
        "at_code": at_code
    }
    
    # Only expand if it's a diagnosis or symptom or condition
    if category in ['diagnosis', 'symptom', 'condition']:
        for prefix in PREFIXES:
            expanded_term = f"{prefix} {base_term}"
            expansions[expanded_term] = {
                "archetype_id": archetype_id,
                "source": "prefix_expansion",
                "at_code": at_code
            }
    return expansions

def parse_opt(opt_path):
    tree = ET.parse(opt_path)
    root = tree.getroot()
    ns = {'ns': 'http://schemas.openehr.org/v1'}
    
    marker_dict = {}
    marker_lookup = {}
    
    # Find all nodes that have an archetype_id
    for arch_node in root.findall('.//ns:archetype_id/..', ns):
        arch_id_node = arch_node.find('ns:archetype_id/ns:value', ns)
        if arch_id_node is None: continue
            
        archetype_id = arch_id_node.text
        category = get_category_for_archetype(archetype_id)
        
        if category not in marker_dict:
            marker_dict[category] = {}
            
        # Parse term definitions inside this archetype node
        for term_def in arch_node.findall('.//ns:term_definitions', ns):
            code = term_def.attrib.get('code')
            
            text_node = term_def.find('./ns:items[@id="text"]', ns)
            desc_node = term_def.find('./ns:items[@id="description"]', ns)
            
            if text_node is not None and text_node.text:
                term_text = text_node.text.lower().strip()
                # Skip very short or generic terms
                if len(term_text) <= 2 or term_text in ['tree', 'history', 'any event']:
                    continue
                    
                expansions = generate_expansions(term_text, archetype_id, code, category)
                marker_dict[category].update(expansions)
                
                # Also populate marker_lookup
                for exp_term, exp_data in expansions.items():
                    # For lookup, we map term -> category info
                    marker_lookup[exp_term] = {
                        "category": category,
                        "archetype_id": archetype_id,
                        "source": exp_data["source"]
                    }
                    
            if desc_node is not None and desc_node.text:
                desc_text = desc_node.text.lower().strip()
                # A very rudimentary way to extract description keywords or just use the description text
                # We'll just add it as a 'term_definition' source
                if len(desc_text) > 5 and len(desc_text.split()) < 5:
                    marker_dict[category][desc_text] = {
                        "archetype_id": archetype_id,
                        "source": "term_definition",
                        "at_code": code
                    }
                    marker_lookup[desc_text] = {
                        "category": category,
                        "archetype_id": archetype_id,
                        "source": "term_definition"
                    }
                
    return marker_dict, marker_lookup

if __name__ == "__main__":
    import sys
    opt_file = r'c:\Users\madha\Desktop\OpenEHR Project\Vital signs.opt'
    if len(sys.argv) > 1:
        opt_file = sys.argv[1]
        
    m_dict, m_lookup = parse_opt(opt_file)
    
    with open(r'c:\Users\madha\Desktop\OpenEHR Project\marker_dictionary.json', 'w', encoding='utf-8') as f:
        json.dump(m_dict, f, indent=2)
        
    with open(r'c:\Users\madha\Desktop\OpenEHR Project\marker_lookup.json', 'w', encoding='utf-8') as f:
        json.dump(m_lookup, f, indent=2)
        
    print(f"Generated {len(m_dict)} categories in marker_dictionary.json")
    print(f"Generated {len(m_lookup)} terms in marker_lookup.json")
