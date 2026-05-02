import json
from collections import defaultdict

def load_advanced_marker_lookup(filepath):
    marker_dict = defaultdict(list)
    archetype_map = {}
    snomed_map = {}
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        for term, meta in data.items():
            category = meta.get('category', 'unknown')
            marker_dict[category].append(term)
            if 'archetype_id' in meta:
                archetype_map[term] = meta['archetype_id']
            if 'at_code' in meta:
                snomed_map[term] = meta['at_code']
                
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        
    return dict(marker_dict), archetype_map, snomed_map

class HybridLLMJudge:
    def __init__(self, use_llm=False):
        self.use_llm = use_llm
        
    def judge_entities(self, text, entities):
        # Dummy pass-through if use_llm is False or LLM is unavailable
        return entities
