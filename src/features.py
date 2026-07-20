import re
import spacy
from spacy.cli import download

class FeatureExtractor:
    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            download("en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")

    def extract_regex_features(self, text):
        features = {}
        
        nic_pattern = r'\b([0-9]{9}[vVxX]|[0-9]{12})\b'
        features['has_nic'] = 1 if re.search(nic_pattern, text) else 0
        
        card_pattern = r'\b(?:\d[ -]*?){13,16}\b'
        features['has_credit_card'] = 1 if re.search(card_pattern, text) else 0
        
        phone_pattern = r'\b(?:\+?94|0)[7-9][0-9]{8}\b|\b(?:\+?1[-. ]?)?\(?([0-9]{3})\)?[-. ]?([0-9]{3})[-. ]?([0-9]{4})\b'
        features['has_phone'] = 1 if re.search(phone_pattern, text) else 0
        
        return features

    def extract_ner_counts(self, text):
        if not text:
            return {"count_person": 0, "count_org": 0, "count_gpe": 0}
            
        doc = self.nlp(text)
        person_count = 0
        org_count = 0
        gpe_count = 0
        
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                person_count += 1
            elif ent.label_ == "ORG":
                org_count += 1
            elif ent.label_ == "GPE":
                gpe_count += 1
                
        return {
            "count_person": person_count,
            "count_org": org_count,
            "count_gpe": gpe_count
        }

    def get_combined_features(self, text):
        regex_feats = self.extract_regex_features(text)
        ner_feats = self.extract_ner_counts(text)
        
        combined = {**regex_feats, **ner_feats}
        return combined
