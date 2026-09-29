# On-Premise Legal Contract Entity Extraction Pipeline (spaCy v3 + Legal-BERT)

An end-to-end, production-grade NLP pipeline designed to extract structured entities from unstructured legal contracts on-premise. By pairing **spaCy v3** with **Legal-BERT** (`nlpaueb/legal-bert-base-uncased`), **Document Zoning**, and a **Hybrid Post-Processing Engine** (`EntityRuler` + heuristics), this pipeline identifies and prioritizes primary contract holders while maintaining complete data privacy and running efficiently on local CPU infrastructure.

---

## 📌 Target Entity Ontology

The system extracts and categorizes six core contract attributes:

| Entity Tag | Description | Example |
| --- | --- | --- |
| `PARTY_ORG` | Organizations entering into the contract | *Acme Global Solutions Inc.* |
| `AGREEMENT` | Title or formal type of the legal document | *Mutual Non-Disclosure Agreement* |
| `EFFECTIVE_DATE` | Execution or start date of the agreement | *October 15, 2024* |
| `EXPIRATION_DATE` | Date when the agreement terminates | *October 14, 2026* |
| `GPE` | Governing laws, jurisdictions, or states | *Delaware*, *Commonwealth of Massachusetts* |
| `PARTY_ADDRESS` | Principal place of business or residence | *742 Evergreen Terrace, Springfield* |

---

## 📁 Repository Directory Layout

```text
Legal Mix/
├── data/                            # Raw and processed datasets
│   ├── CUADv1.json                  # Downloaded CUAD corpus (Zenodo)
│   ├── train.spacy                  # Processed training binary (80% split)
│   └── val.spacy                    # Processed validation binary (20% split)
│
├── scripts/                         # Local execution scripts (Windows 10)
│   ├── download_cuad.py             # Resilient Zenodo ingestion script
│   ├── convert_cuad_to_spacy.py     # Conversion, chunking, & span-filtering engine
│   ├── zone_contract.py             # Document zoning engine (Preamble + Signatures)
│   ├── postprocess.py               # Hybrid EntityRuler & heuristic verification
│   └── inference.py                 # CPU inference pipeline & JSON formatter
│
├── notebooks/                       # Google Colab GPU training workflows
│   └── CUAD_Dataset_Training.ipynb  # Transformer fine-tuning & config generator
│
├── model/                           # Exported model artifacts
│   └── model-best/                  # Best fine-tuned spaCy transformer pipeline
│
├── requirements.txt                 # Python dependencies
└── README.md                        # Documentation

```

---

## 🧠 System Architecture & Design Strategy

Extracting structured data from commercial legal contracts presents two major challenges:

1. **High Entity Noise:** Contracts mention dozens of secondary organizations (governing bodies, banks, third-party licensors, witnesses). Standard NER extracts all of them without distinguishing priority.
2. **Contextual & Formatting Variety:** Entity names often appear without explicit keyphrase labels (e.g., without `"Organization Name:"` or `"Address:"`).

### Architecture Flow

```
   Raw Contract File (.txt / .pdf)
                │
                ▼
  [ 1. Document Zoning Engine ] ──► Isolates Preamble (0-1500 chars) & Signature Block
                │
                ▼
  [ 2. Fine-Tuned Legal-BERT ]  ──► Contextual Entity Recognition (spaCy v3)
                │
                ▼
  [ 3. Rule-Based EntityRuler ] ──► Regex enforcement for zip codes, states, suffixes
                │
                ▼
  [ 4. Cross-Zone Verifier ]    ──► Promotes entities appearing in Preamble + Signatures
                │
                ▼
      Structured JSON Output

```

---

## 🛠️ Step-by-Step Execution & Engineering Challenge Log

This section records every step, bug, edge-case failure, and architectural fix implemented during development.

### Phase 1: Local Environment & Workspace Setup (Step 0)

Development was split into two environments:

* **Local Machine:** Windows 10 (CPU-only), Python 3.11/3.13, virtual environment (`spacy_legal_env`), and `Label Studio` for local annotation inspection.
* **Training Machine:** Google Colab (T4 GPU, 15GB VRAM) running PyTorch with CUDA acceleration (`cuda12x`).

### Phase 2: Dataset Ingestion Battles (CUAD Dataset)

To avoid spending hundreds of hours manually tagging text, we utilized the **CUAD (Contract Understanding Atticus Dataset)**. Ingesting this dataset required overcoming multiple package failures:

* **Issue 1: Hugging Face API Script Deprecation (`HfUriError`)**
* *Error:* Running `load_dataset("cuad")` threw `HfUriError: Invalid HF URI... Repository id must be 'namespace/name'`.
* *Root Cause:* Modern Hugging Face `datasets` versions disabled un-namespaced dataset calls and deprecated remote Python execution scripts.
* *Attempted Fix:* Switching to `theatticusproject/cuad-qa` triggered Windows developer mode symlink warnings (`huggingface_hub` cache errors).


* **Fix Applied:** Built a standalone fallback downloader (`download_cuad.py`) using `urllib.request` to pull `CUAD_v1.zip` (~105 MB) directly from the official Zenodo repository.
* **Issue 2: Zenodo Archive Path Extraction Mismatch**
* *Error:* `FileNotFoundError: [Errno 2] No such file or directory: 'CUAD_v1\CUADv1.json'`.
* *Root Cause:* Unzipping Zenodo archives behaves differently across OS environments (extracting files into the root working directory rather than creating a `CUAD_v1/` folder).
* *Fix Applied:* Wrote a recursive search helper function (`find_json_file()`) that scans all local subdirectories to locate `CUADv1.json` regardless of zip extraction structure.



### Phase 3: Dataset Conversion & Span Filtering (`convert_cuad_to_spacy.py`)

Converting `CUADv1.json` into binary `.spacy` format (`train.spacy` and `val.spacy`) introduced critical NLP bugs:

* **Issue 1: Overlapping Entity Span Crashes**
* *Error:* spaCy's standard NER parser threw fatal exceptions when entity spans overlapped (e.g., `"Acme Corp"` tagged as `PARTY_ORG` inside `"Acme Corp, a Delaware company"` tagged as `PARTY_BLOCK`).
* *Fix Applied:* Integrated `spacy.util.filter_spans()` inside the conversion loop. This retains the longest valid entity span and silently drops conflicting sub-spans.


* **Issue 2: Character-to-Token Offset Misalignment**
* *Error:* Character start/end positions in raw JSON spans occasionally landed inside word tokens (e.g., trailing commas or quotes).
* *Fix Applied:* Enforced `alignment_mode="contract"` on `doc.char_span()`, snapping entity boundaries tightly to whole token limits.


* **Issue 3: The 512-Token BERT Sequence Limit (`1618 > 512`)**
* *Error:* Contracts in CUAD often span 1,500+ words. Passing entire contract bodies into `Legal-BERT` triggered warnings (`Token indices sequence length is longer than maximum sequence length (1618 > 512)`), causing memory allocation failures during validation.
* *Fix Applied:* Implemented a sentence-based sliding window (`spacy` `sentencizer`) in `convert_cuad_to_spacy.py`. The converter breaks raw documents into chunks of $\le 300$ words, recalibrates relative entity offsets for each chunk, and saves clean, truncated `Doc` objects into `DocBin`.



---

### Phase 4: Google Colab GPU Fine-Tuning & Config Debugging

Fine-tuning `nlpaueb/legal-bert-base-uncased` on GPU required solving multiple spaCy configuration issues:

#### 1. Configuration Key Errors

* **`DuplicateOptionError: option 'grad_factor' in section 'components.transformer.model' already exists`**
* *Cause:* Manual string injection appended `grad_factor = 2` into `config.cfg` when the key was already defined.


* **`unexpected argument: 'max_words'`**
* *Cause:* Legacy spaCy v2 corpus reader arguments were deprecated in spaCy v3.


* **`Config error for 'spacy-transformers.TransformerModel.v3': unexpected argument: 'grad_factor'`**
* *Cause:* `spacy-transformers` v1.3+ moved gradient accumulation from the model architecture block into `[training].accumulate_gradient`.



#### The Fix: Programmatic Config Generation

Instead of manual text editing or string replaces, we wrote a Python initialization block using spaCy's native `confection` library:

```python
import spacy
from confection import Config

# Generate clean base config
!python -m spacy init config config.cfg --lang en --pipeline transformer,ner --optimize accuracy --gpu --force

# Load via Confection
config = Config().from_disk("config.cfg")

# Set Legal-BERT backbone
config["components"]["transformer"]["model"]["name"] = "nlpaueb/legal-bert-base-uncased"

# Manage gradient accumulation globally
config["training"]["accumulate_gradient"] = 3

# Fix learning rate schedule (Avoids 0.0 learn rate bug)
config["training"]["optimizer"]["learn_rate"] = {
    "@schedules": "warmup_linear.v1",
    "initial_rate": 2e-05,
    "warmup_steps": 250,
    "total_steps": 20000
}

# Add gradient clipping to prevent loss spikes
config["training"]["optimizer"]["grad_clip"] = 1.0

# Save updated config
config.to_disk("config.cfg")

```

#### 2. Diagnosing & Solving the `ENTS_F = 0.00` & Loss Explosion Issue

During early training runs, the logs showed catastrophic behavior:

```text
#       LOSS TRANS...  LOSS NER  ENTS_F  ENTS_P  ENTS_R  SCORE 
200      125831.45   7140.52    0.00    0.00    0.00    0.00
400         189.99   1996.03    0.00    0.00    0.00    0.00
800        6055.47   2854.68    0.00    0.00    0.00    0.00

```

* **Root Cause 1 (`Initial learn rate: 0.0`):** The header logs revealed `Initial learn rate: 0.0`. The `warmup_linear` schedule lacked an explicit `total_steps` definition, causing the math to collapse learning rate to zero. The model weights remained completely frozen while gradient accumulation loss spiked past $125,000$.
* **Root Cause 2 (Validation Truncation Mismatch):** Unchunked validation files passed sequences of 566–1,618 tokens to `val.spacy`. During validation evaluation, spaCy truncated these sequences, misaligning character offsets and causing the strict scorer to record an F1 score of `0.00`.
* **The Resolution:**
1. Re-chunked `val.spacy` to strict $<300$ token blocks.
2. Forced `initial_rate = 2e-5` with `grad_clip = 1.0`.
3. Set `accumulate_gradient = 3`.



Upon applying these fixes, loss metrics stabilized and `ENTS_F` (F1 Score) climbed steadily past **0.82+ (82%)**.

---

### Phase 5: Post-Processing & Entity Linking Engine (`postprocess.py`)

After exporting `model-best` from Google Colab to local Windows CPU, we built two post-processing components:

1. **Document Zoning Engine (`zone_contract.py`)**
* **Preamble Zone:** Extracts the first 1,500 characters of the document.
* **Signature Zone:** Extracts the last 1,500 characters of the document.


2. **Hybrid EntityRuler & Priority Matcher (`postprocess.py`)**
* Applies regular expressions for standard patterns (e.g., zip codes, state abbreviations, corporate suffixes like `LLC`, `Inc.`, `Corp.`).
* **Cross-Zone Verification:** Compares `PARTY_ORG` predictions between the Preamble and Signature Block. If an organization is present in both zones, it is flagged as a **Primary Contract Holder**. Organizations found only in body text (e.g., banks or third-party software vendors) are demoted to `mentioned_entities`.



---

## 🚀 Installation & Local CPU Execution Guide

### 1. Environment Setup (Windows 10 Local)

```cmd
# Create virtual environment
python -m venv spacy_legal_env
spacy_legal_env\Scripts\activate

# Install dependencies
pip install spacy[transformers] torch datasets urllib3 fastapi uvicorn

```

### 2. Download CUAD & Generate Binary Files

```cmd
# Download CUADv1.json from Zenodo
python scripts/download_cuad.py

# Convert raw data into chunked .spacy binaries
python scripts/convert_cuad_to_spacy.py

```

### 3. Fine-Tune on Google Colab (T4 GPU)

1. Upload `data/train.spacy` and `data/val.spacy` to Google Drive.
2. Open `notebooks/CUAD_Dataset_Training.ipynb` in Colab.
3. Enable GPU Runtime (**Runtime ➔ Change runtime type ➔ T4 GPU**).
4. Run all notebook cells to execute `spacy train config.cfg`.
5. Download the output folder `contract_model/model-best` to your local `./model/model-best` directory.

### 4. Run Local Inference (Windows CPU)

```python
from scripts.inference import ExtractContractEntities

# Initialize pipeline with local model
extractor = ExtractContractEntities(model_path="./model/model-best")

contract_text = """
This MUTUAL NON-DISCLOSURE AGREEMENT is entered into on October 15, 2024,
by and between Acme Global Solutions Inc., a Delaware corporation located at 
742 Evergreen Terrace, Springfield ("Disclosing Party"), and Beta Enterprises LLC ("Receiving Party").
"""

result = extractor.process(contract_text)
print(result)

```

---

## 📄 Output JSON Schema

```json
{
  "document_metadata": {
    "agreement_type": "MUTUAL NON-DISCLOSURE AGREEMENT",
    "effective_date": "October 15, 2024",
    "governing_law": "Delaware"
  },
  "primary_contract_holders": [
    {
      "name": "Acme Global Solutions Inc.",
      "alias": "Disclosing Party",
      "address": "742 Evergreen Terrace, Springfield"
    },
    {
      "name": "Beta Enterprises LLC",
      "alias": "Receiving Party",
      "address": null
    }
  ],
  "mentioned_entities": []
}

```