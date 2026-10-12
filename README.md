# Medical RAG Assistant: Clinical Reasoning with Dual-Index Semantic Retrieval

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FAISS](https://img.shields.io/badge/Vector%20Search-FAISS-green.svg)](https://github.com/facebookresearch/faiss)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red.svg)](https://streamlit.io/)
[![Gemini](https://img.shields.io/badge/LLM-Gemini%202.0%20Flash-orange.svg)](https://ai.google.dev/)

An enterprise-grade Retrieval-Augmented Generation (RAG) system engineered for complex clinical diagnostic reasoning. The system retrieves grounded medical evidence from structured MIMIC-IV clinical archives through a dual-index vector pipeline and generates verifiable, safety-guarded clinical diagnoses using Gemini 2.0 and local LLMs.

---

## Architecture Overview

Traditional single-index RAG systems often retrieve generic medical literature that lacks patient context, or conversely, match raw patient cases without clinical guidelines. This system introduces a **Dual-Index Semantic Architecture** that retrieves both simultaneously.

```
                              [ Incoming Patient Query ]
                             (Symptoms, Vitals, Lab Tests)
                                          |
                                          v
                         [ nomic-embed-text-v1.5 Encoder ]
                           (768-dimensional dense vector)
                                          |
                                          v
                             [ L2 Vector Normalization ]
                                          |
                   +----------------------+----------------------+
                   |                                             |
                   v                                             v
        [ Medical KB Index ]                          [ Patient Cases Index ]
      (FAISS IndexFlatIP: 768D)                     (FAISS IndexFlatIP: 768D)
                   |                                             |
                   v                                             v
       Top-2 Clinical Guidelines                      Top-2 Similar Patient Cases
       (cosine similarity score)                      (cosine similarity score)
                   |                                             |
                   +----------------------+----------------------+
                                          |
                                          v
                            [ Augmented Context Block ]
                        - Clinical Guidelines (Relevance: 0.88)
                        - Historical Patient Case (Relevance: 0.85)
                                          |
                                          v
                       [ Safety-Guarded Clinical Prompt ]
                        - Structured diagnostic template
                        - Strict evidence grounding instruction
                        - Insufficient data safety refusal gate
                                          |
                                          v
                            [ Google Gemini 2.0 Flash ]
                            (or Local DeepSeek / Ollama)
                                          |
                                          v
                             [ Structured Output ]
                        1. Extracted Evidence Summary
                        2. Confirmed Diagnosis
                        3. Bulleted Clinical Reasoning
```

---

## Why This Architecture Was Chosen

### 1. Dual-Index Retrieval (Guidelines + Historical Cases)
- **Problem:** When diagnosing complex conditions (e.g., atypical NSTEMI in diabetic patients), guidelines give general criteria, while historical case charts provide real-world symptom presentations.
- **Solution:** We maintain two decoupled FAISS vector spaces:
  - `kb_index.faiss`: Curated medical knowledge base (`medicalKB`) establishing evidence standards.
  - `patient_index.faiss`: Granular case histories (`patient_case`) containing troponin levels, ECG findings, and verified discharge diagnoses.
- Querying both indices concurrently guarantees that generation is anchored by both clinical theory and real patient precedents.

### 2. High-Performance Dense Embeddings (`nomic-embed-text-v1.5`)
- Selected for its 8,192 token context window and strong MTEB clinical retrieval benchmarks.
- Encodes complex multi-paragraph clinical notes into 768-dimensional dense vectors without truncation.

### 3. FAISS `IndexFlatIP` with L2 Normalization (Exact Cosine Similarity)
- Vectors are normalized to unit length (`faiss.normalize_L2`), transforming inner product (dot product) calculations into exact cosine similarity:
  `Cosine Similarity = Dot Product(Query, Document)`
- Provides mathematically exact similarity scoring between `0.0` and `1.0` with sub-millisecond retrieval latency, avoiding approximate search artifacts.

### 4. Anti-Hallucination & Safety Refusal Prompting
- The LLM is explicitly barred from generating diagnoses unsupported by the retrieved documents.
- Includes a safety gate: if evidence similarity scores are insufficient or contradictory, the system is instructed to refuse diagnostic commitment and flag the case for manual physician escalation.

---

## Quantitative & Qualitative Evaluation (NLP Research)

The system includes comprehensive offline evaluation experiments developed in `notebooks/nlp-a04.ipynb`:

1. **Dataset:** Curated subset of the **MIMIC-IV** clinical database containing multi-modal patient admissions, vital signs, cardiac biomarkers (troponin I/T), ECG reads, and discharge summaries.
2. **Relevance Scoring:** Evaluated across multiple clinical query variants (NSTEMI with CAD, 3-Vessel Occlusion, Epigastric presentation with Diabetic Neuropathy). Top-2 retrieval maintained a mean cosine similarity score above `0.82`.
3. **Comparative Generation:** Evaluated zero-shot LLM generation against RAG-grounded generation:
   - **Zero-shot LLM:** Frequently misattributed epigastric symptoms in diabetic patients to gastrointestinal distress alone.
   - **Dual-Index RAG:** Correctly retrieved atypical myocardial infarction guidelines and diabetic neuropathy precedents, identifying acute coronary ischemia and elevated troponin risks.

---

## Interactive Streamlit Interface

The project includes an interactive web dashboard (`app.py` / `localapp.py`):
- **Pre-configured Clinical Scenarios:** Instant test cases covering complex presentations (e.g., NSTEMI with CAD, NSTEMI with 3-Vessel Disease, Atypical Diabetic NSTEMI).
- **Transparency & Inspectability:** Expander panels display the exact retrieved guidelines, historical case inputs, and similarity scores.
- **Privacy First:** Patient identifiers and sensitive headers are stripped before vectorization and API calls.

---

## Quickstart & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/hurairamuzammal/NLP-RAG-medical-LLM.git
cd NLP-RAG-medical-LLM
```

### 2. Set Up Virtual Environment & Dependencies
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Configure Environment Variables
Create a `.env` file in the project root:
```env
GEMINI_API_KEY=your_gemini_api_key_here
```

### 4. Run the Streamlit Application
```bash
streamlit run app.py
```
Or run the local model variant:
```bash
streamlit run localapp.py
```

---

## Technology Stack

- **Framework:** Python 3.10+, Streamlit
- **Embeddings:** Hugging Face `sentence-transformers`, `nomic-ai/nomic-embed-text-v1.5`
- **Vector Engine:** Facebook AI Similarity Search (`faiss-cpu`)
- **LLM Providers:** Google Gemini 2.0 Flash (`google-generativeai`), Local Hugging Face / Ollama
- **Dataset:** MIMIC-IV Clinical Database Extraction
