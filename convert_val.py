import json
import os
import spacy
from spacy.tokens import DocBin

def create_clean_val_dataset(json_path="CUAD_v1\CUAD_v1.json", output_val="val.spacy"):
    print("Loading CUADv1.json to extract clean validation chunks...")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    nlp_sent = spacy.blank("en")
    nlp_sent.add_pipe("sentencizer")

    label_mapping = {
        "Parties": "PARTY_ORG",
        "Document Name": "AGREEMENT",
        "Effective Date": "EFFECTIVE_DATE",
        "Expiration Date": "EXPIRATION_DATE",
        "Governing Law": "GPE"
    }

    doc_bin_val = DocBin()
    contracts = data.get("data", [])
    val_contracts = contracts[int(len(contracts) * 0.8):]  # Last 20% for validation

    valid_chunks_count = 0

    for contract in val_contracts:
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
                        span = doc.char_span(start_pos, start_pos + len(text_ans), label=target_label, alignment_mode="contract")
                        if span is not None:
                            candidate_entities.append(span)

        doc.ents = spacy.util.filter_spans(candidate_entities)

        # Sentence-Chunking: Group sentences into max 300 words
        accumulated_tokens = []
        accumulated_ents = []
        current_length = 0

        for sent in doc.sents:
            if current_length + len(sent) > 300 and accumulated_tokens:
                chunk_doc = spacy.tokens.Doc(nlp_sent.vocab, words=accumulated_tokens)
                chunk_doc.ents = spacy.util.filter_spans(accumulated_ents)
                doc_bin_val.add(chunk_doc)
                valid_chunks_count += 1
                
                accumulated_tokens = []
                accumulated_ents = []
                current_length = 0

            start_offset = current_length
            for token in sent:
                accumulated_tokens.append(token.text)

            for ent in sent.ents:
                rel_start = ent.start - sent.start + start_offset
                rel_end = ent.end - sent.start + start_offset
                accumulated_ents.append(spacy.tokens.Span(doc, rel_start, rel_end, label=ent.label_))

            current_length += len(sent)

    doc_bin_val.to_disk(output_val)
    print(f"Validation dataset saved to {output_val} ({valid_chunks_count} clean chunks under 300 tokens).")

if __name__ == "__main__":
    create_clean_val_dataset()