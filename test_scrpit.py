import spacy
import json

# 1. Load your fine-tuned model checkpoint
model_path = "model-best"  # or "./contract_model/model-best"
print(f"Loading fine-tuned model from {model_path}...")

try:
    nlp = spacy.load(model_path)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# 2. Sample legal contract preamble text
sample_contract = """
MUTUAL NON-DISCLOSURE AGREEMENT

This Mutual Non-Disclosure Agreement ("Agreement") is made and entered into as of October 15, 2024 ("Effective Date"),
by and between Acme Global Solutions Inc., a Delaware corporation having its principal place of business at 
742 Evergreen Terrace, Springfield, OR 97477 ("Disclosing Party"), and Beta LLC, a California limited liability company ("Receiving Party").

This Agreement shall remain in effect until October 14, 2026 ("Expiration Date").
The performance and construction of this Agreement shall be governed by the laws of the State of Delaware.
"""

# 3. Run model inference
doc = nlp(sample_contract)

# 4. Format extracted entities into a clean dictionary
extracted_data = {
    "PARTY_ORG": [],
    "AGREEMENT": [],
    "EFFECTIVE_DATE": [],
    "EXPIRATION_DATE": [],
    "GPE": [],
    "PARTY_ADDRESS": []
}

print("\n" + "=" * 50)
print("EXTRACTED ENTITIES FROM CONTRACT")
print("=" * 50)

if not doc.ents:
    print("[!] No entities extracted! (Model returned empty predictions)")
else:
    for ent in doc.ents:
        print(f"[{ent.label_:<15}] -> {ent.text}")
        if ent.label_ in extracted_data:
            extracted_data[ent.label_].append(ent.text)

print("\n" + "=" * 50)
print("STRUCTURED JSON OUTPUT")
print("=" * 50)
print(json.dumps(extracted_data, indent=2))