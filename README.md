# SENTINEL

### Edge-Native Clinical Consultation Node for Regulated Mental Health Services
#### Offline-First · Powered by Gemma 4 (e4b) · Tamper-Evident SQLCipher Audit Chain · Microsoft Presidio PHI Anonymization

---

> [!IMPORTANT]  
> **100% On-Premises / Air-Gapped Operation.** SENTINEL runs entirely on local edge hardware. It requires no external API connections, sends zero bytes over the network, and enforces strict patient privacy (HIPAA/GDPR) via offline PHI scrubbing before write-ahead logging.

---

## 🌟 The Problem & The SENTINEL Solution

Critical technical and clinical environments (such as remote clinics, disaster zones, or air-gapped hospital networks) rely on complex clinical guidelines like the **WHO mhGAP Intervention Guide (v2.0)**. 

Existing cloud-AI solutions are unsuitable for these environments because:
1. **Network Connectivity:** Operators or clinicians work in environments with poor or zero internet access.
2. **Data Privacy (HIPAA/GDPR):** Uploading patient data to third-party APIs violates data sovereignty laws and patient confidentiality.
3. **Safety & Hallucinations:** Default RAG pipelines can hallucinate treatment advice or fail to detect patient crisis signals.

**SENTINEL solves this at the edge:** It provides a lightweight, highly accurate, fully offline clinical support assistant that processes queries, retrieves guidelines using hybrid FTS+vector search, streams answers locally, validates claim faithfulness via Natural Language Inference (NLI), and signs every reasoning step to an encrypted, tamper-evident audit ledger.

---

## 🛠 Tech Stack & Architecture Rationale

| Component | Technology | Technical Rationale |
| :--- | :--- | :--- |
| **Local LLM Engine** | **Gemma 4 (e4b)** (via Ollama) | **Effective 4-Billion Footprint:** Uses Per-Layer Embeddings (PLE) and vocabulary sharing to deliver 8B-level clinical reasoning with a 4.5B-level compute footprint (~9.6 GB RAM). Native **128K context window** prevents truncation of detailed WHO guidelines and comorbid session history. |
| **Vector & Text Store** | **LanceDB** (`>=0.13.0`) | **Single-Store Vector+FTS:** Integrates Rust-backed Tantivy FTS (BM25) and IVF-HNSW-SQ vector index inside a single file-backed database. Ensures zero dual-store synchronization latency and atomic transactions. |
| **Layout Ingestion** | **Docling** | Layout-aware document parsing. Extracted structure normalizes headings, restores nested clinical lists, and runs a custom **heuristics-based decision tree extractor** to chunk clinical flowcharts as atomic units. |
| **Sentence Splitter** | **scispaCy** (`en_core_sci_sm`) | Specifically trained on medical abbreviations (e.g., *b.i.d.*, *q.d.*, *Dr.*, *ICD-10 codes*) to split generated answers into sentence tokens without breaking on clinical abbreviations. Falls back to NLTK. |
| **Faithfulness Gate** | **NLI ONNX** (`nli-MiniLM2-L6-H768`) | Runs batched Natural Language Inference comparing every generated sentence (hypothesis) to retrieved WHO chunks (premise) to calculate entailment. Rejects hallucinated claims before response transmission. |
| **PHI Scrubbing** | **Microsoft Presidio** (Offline) | Extracts and masks PII/PHI (names, dates, locations, phone numbers) locally using a pre-downloaded spaCy NER model, preventing any patient data from entering logs. |
| **Audit Chain Ledger** | **SQLCipher WAL** | Cryptographically secures logs at-rest using AES-256 database encryption. Chaining via SHA-256 hashes and Ed25519 signatures guarantees tamper detection. |
| **FastAPI Backend** | **FastAPI + slowapi** | High-performance, local-only server with client-bound rate-limits (5 req/min) to prevent concurrency starvation. Streams tokens live using Server-Sent Events (SSE). |
| **Operator UI** | **Vanilla HTML5 / JS** | Zero Node.js runtime. Meets WCAG 2.1 AA accessibility standards (4.5:1 contrast, dark mode, keyboard navigation, full ARIA roles). |

---

## 📐 System Pipeline Architecture

```
                                  [ CLINICIAN QUERY ]
                                           │
                                           ▼
                                   Step 0: PHI Scrub (Microsoft Presidio)
                                           │
                                           ▼
                                 Step 1: Crisis Detector
                                  ├── Tier-1 (Hard Block) ──► Immediate WHO Emergency Response
                                  └── Tier-2 (Constrained)
                                           │
                                           ▼
                               Step 2: Intent Classifier (Gemma 4)
                                           │
                                           ▼
                               Step 3: Hybrid Retrieval (LanceDB)
                                  ├── Text Search (Tantivy BM25) ──┐
                                  └── Vector Search (Cosine ANN) ──┴─► RRF Fusion (Top 20)
                                           │
                                           ▼
                                Step 4: Reranking (MiniLM CrossEncoder)
                                           │
                                           ▼
                              Step 5: Clinical Alerts Validation (Note/Caution)
                                           │
                                           ▼
                                 Step 6: LLM Synthesis (Gemma 4) ──► SSE Token Stream
                                           │
                                           ▼
                              Step 7: Sentence Tokenizer (scispaCy)
                                           │
                                           ▼
                             Step 8: Faithfulness Check (MiniLM NLI)
                                  ├── Validated (Entailment >= 0.75) ──► Complete Stream
                                  └── Hallucinated (Contradiction/Neutral)
                                           │
                                           ▼
                              Step 9: Refine (Max 3 iterations) or Escalate
                                           │
                                           ▼
                             Step 10: Log to Encrypted Audit Ledger (SQLCipher)
```

---

## ⚡ Quick Start

### Prerequisites
* **OS:** macOS (Apple Silicon recommended) or Linux.
* **Inference:** [Ollama](https://ollama.ai) installed and running.
* **Python:** Version 3.11+.
* **Package Manager:** `uv` installed (`curl -LsSf https://astral.sh/uv/install.sh | sh`).

### 1. Model & Dependency Bootstrap
Run the bootstrap script on an internet-connected machine to pre-cache the models (Ollama weights, rerankers, NLI, spaCy language packs, and python wheels):
```bash
chmod +x scripts/bootstrap_offline.sh
./scripts/bootstrap_offline.sh
```

### 2. Initialize Cryptographic Keys
Generate HMAC signing and SQLCipher encryption keys and store them securely in the OS Keychain (`keyring`):
```bash
export OLLAMA_KEEP_ALIVE=-1
ollama serve &
uv run python scripts/audit_key_init.py
```

### 3. Verify Environment
Ensure all index files, model caches, Keychain items, and database locks are configured correctly:
```bash
uv run python scripts/verify_environment.py
```

### 4. Ingest WHO mhGAP Guidelines
Place the target PDF document (e.g., `mhgap_ig_v2.0_2016.pdf`) in the `data/corpus/` directory, then run the CPU-optimized layout parser:
```bash
uv run sentinel ingest data/corpus/mhgap_ig_v2.0_2016.pdf
```

### 5. Launch the Local Service
Start the FastAPI server:
```bash
uv run uvicorn api.server:app --host 127.0.0.1 --port 8000
```
Access the operator dashboard at **[http://127.0.0.1:8000](http://127.0.0.1:8000)**.

---

## 📊 Evaluation & Verification Harness

SENTINEL contains an automated evaluation script that validates three core safety metrics:
1. **Crisis Detection Precision/Recall:** Tested against 100 balanced crisis scenarios (balanced across Tier-1, Tier-2, and NONE). Target: **Tier-1 Recall >= 99%**, **Precision >= 85%**.
2. **Adversarial Probes:** Evaluates PHI scrubbing success (identifying and masking hidden patient data), unsupported medication blocking, and non-English warnings.
3. **Golden Q&A Dataset:** Tests 50 mhGAP QA pairs across MNS condition codes to verify intent routing accuracy and RAG citation recall.

To execute the benchmark harness:
```bash
uv run python eval/run_eval.py
```

---

## 🛡 Security, Safety, and Compliance

* **Audit Chain Integrity:** Every action generates a SHA-256 linked log entry signed with an HMAC key stored in the OS Keychain. You can run the audit verifier at any time to check for database tampering or log deletions:
  ```bash
  uv run sentinel audit-verify
  ```
* **Strict PHI-Free Audit Logs:** The audit database stores the SHA-256 hash of the original query (`original_query_hash`) and a max-50 character de-identified text snippet (`scrubbed_query_preview`). Patient names, phone numbers, and dates are completely scrubbed before writing.
* **Crisis Safety Screen:** If a clinician's query triggers a Tier-1 keyword or semantic match, RAG synthesis is completely bypassed, and a pre-written, WHO-aligned crisis action protocol (suicide/self-harm mitigation instructions) is returned immediately.
