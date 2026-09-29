import os
import json
import zipfile
import urllib.request
import spacy
from spacy.tokens import DocBin

ZENODO_CUAD_ZIP_URL = "https://zenodo.org/records/4595826/files/CUAD_v1.zip?download=1"
ZIP_FILE_NAME = "CUAD_v1.zip"

def find_json_file(filename="CUAD_v1.json", search_path="."):
    for root, dirs, files in os.walk(search_path):
        if filename in files:
            return os.path.join(root, filename)
    return None

def download_and_extract_cuad():
    target_path = find_json_file("CUAD_v1.json")
    if not target_path:
        if not os.path.exists(ZIP_FILE_NAME):
            print("Downloading CUAD dataset (~105MB) from Zenodo...")
            req = urllib.request.Request(
                ZENODO_CUAD_ZIP_URL, 
                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
            )
            with urllib.request.urlopen(req) as response, open(ZIP_FILE_NAME, 'wb') as out_file:
                out_file.write(response.read())
            print("Download complete!")
            
        print("Extracting dataset archive...")
        with zipfile.ZipFile(ZIP_FILE_NAME, 'r') as zip_ref:
            zip_ref.extractall(".")
        print("Extraction complete!")
        
        target_path = find_json_file("CUAD_v1.json")
        if not target_path:
            raise FileNotFoundError("CUAD_v1.json could not be located.")
            
    print(f"Located target file: {target_path}")
    return target_path

def convert_cuad_to_spacy(output_train="train.spacy", output_val="val.spacy", train_split_ratio=0.8):
    target_json_path = download_and_extract_cuad()
    
    print("Loading and parsing CUAD_v1.json...")
    with open(target_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Load sentence splitter
    nlp_sent = spacy.blank("en")
    nlp_sent.add_pipe("sentencizer")
    
    label_mapping = {
        "Parties": "PARTY_ORG",
        "Document Name": "AGREEMENT",
        "Effective Date": "EFFECTIVE_DATE",
        "Expiration Date": "EXPIRATION_DATE",
        "Governing Law": "GPE"
    }

    doc_bin_train = DocBin()
    doc_bin_val = DocBin()

    contracts = data.get("data", [])
    total_contracts = len(contracts)
    train_limit = int(total_contracts * train_split_ratio)
    
    processed_count = 0

    print(f"Processing {total_contracts} contracts with sentence-chunking...")

    for idx, contract in enumerate(contracts):
        paragraphs = contract.get("paragraphs", [])
        if not paragraphs:
            continue
            
        context_text = paragraphs[0].get("context", "")
        if not context_text.strip():
            continue

        doc = nlp_sent(context_text)
        candidate_entities = []

        for qa in paragraphs[0].get("qas", []):
            question_text = qa.get("question", "")
            target_label = None
            for key, label in label_mapping.items():
                if key.lower() in question_text.lower():
                    target_label = label
                    break
            
            if target_label:
                for answer in qa.get("answers", []):
                    text_ans = answer.get("text", "")
                    start_pos = answer.get("answer_start", -1)
                    
                    if start_pos != -1 and text_ans:
                        end_pos = start_pos + len(text_ans)
                        span = doc.char_span(start_pos, end_pos, label=target_label, alignment_mode="contract")
                        if span is not None:
                            candidate_entities.append(span)

        doc.ents = spacy.util.filter_spans(candidate_entities)

        # CHUNKING LOGIC: Break long document into smaller sentence groups to avoid transformer OOM
        accumulated_tokens = []
        accumulated_ents = []
        current_length = 0

        for sent in doc.sents:
            # If current chunk exceeds ~350 words, finalize and add to DocBin
            if current_length + len(sent) > 350 and accumulated_tokens:
                chunk_doc = spacy.tokens.Doc(nlp_sent.vocab, words=accumulated_tokens)
                chunk_doc.ents = spacy.util.filter_spans(accumulated_ents)
                if idx < train_limit:
                    doc_bin_train.add(chunk_doc)
                else:
                    doc_bin_val.add(chunk_doc)
                
                accumulated_tokens = []
                accumulated_ents = []
                current_length = 0

            # Append sentence tokens and offset entities
            start_offset = current_length
            for token in sent:
                accumulated_tokens.append(token.text)
            
            for ent in sent.ents:
                rel_start = ent.start - sent.start + start_offset
                rel_end = ent.end - sent.start + start_offset
                accumulated_ents.append(spacy.tokens.Span(doc, rel_start, rel_end, label=ent.label_))
                
            current_length += len(sent)

        # Add remaining tokens
        if accumulated_tokens:
            chunk_doc = spacy.tokens.Doc(nlp_sent.vocab, words=accumulated_tokens)
            try:
                chunk_doc.ents = spacy.util.filter_spans(accumulated_ents)
                if idx < train_limit:
                    doc_bin_train.add(chunk_doc)
                else:
                    doc_bin_val.add(chunk_doc)
            except Exception:
                pass

        processed_count += 1
        if (idx + 1) % 50 == 0:
            print(f"Processed {idx + 1}/{total_contracts} contracts...")

    doc_bin_train.to_disk(output_train)
    doc_bin_val.to_disk(output_val)

    print("\n--- Chunked Conversion Finished ---")
    print(f"Saved chunked files: {output_train} and {output_val}")

if __name__ == "__main__":
    convert_cuad_to_spacy()