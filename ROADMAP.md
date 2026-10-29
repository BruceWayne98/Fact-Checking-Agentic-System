# 🗺️ TRUST Agents — Implementation Roadmap

> **Goal:** Re-implement the TRUST Agents paper (arXiv:2604.12184) from scratch, learning each component deeply as you build it.
> **Stack:** Python · LangGraph · Ollama (local LLM) · spaCy · BM25 · FAISS
> **Style:** Learn-by-building — every phase teaches you the *why* before the *how*

---

## Reading Guide

Each phase has three sections:
- 📚 **Learn** — what concept this phase is about and why it matters
- 🔨 **Build** — what you will implement
- ✅ **Checkpoint** — how to verify you've done it correctly

Phases are sequential. Don't skip — each phase's output feeds the next.

---

## Phase 0 — Environment Setup

### 📚 Learn
Before writing a single agent, you need a working local AI stack. This phase teaches you:
- What **Ollama** is and how local LLM inference works (no API key, no cloud, runs on your machine)
- What **LangChain / LangGraph** is and how they differ:
  - LangChain = building blocks (LLM wrappers, prompt templates, chains)
  - LangGraph = state machine that orchestrates those blocks as a graph of nodes/edges
- How Python virtual environments keep dependencies isolated

### 🔨 Build
- [x] Install Python 3.11+ (Python 3.14.7 installed)
- [x] Create a virtual environment (`python -m venv .venv`)
- [x] Install core dependencies (`langgraph`, `langchain`, `spacy`, `faiss-cpu`, `rank-bm25`, `sentence-transformers`, `datasets`, `scikit-learn`, etc.)
- [x] Download spaCy English model: `en_core_web_sm`
- [x] Scaffold the project folder structure
- [x] Write `config.py` — model name, paths, confidence thresholds
- [x] Install Ollama and pull model: `ollama pull llama3.2` (or start Ollama desktop service)

### ✅ Checkpoint
Run this sanity check — all core dependencies verified:
```
[OK] LangGraph
[OK] spaCy (core_web_sm)
[OK] rank-bm25
[OK] faiss-cpu (v1.15.1)
[OK] sentence-transformers
[OK] datasets & scikit-learn
[OK] ollama client library
```

---

## Phase 1 — Understanding the LIAR Dataset

### 📚 Learn
Before building any detector, you must understand what you're detecting against. This phase teaches you:
- What the **LIAR benchmark** is: 12,836 short political statements from PolitiFact, labeled across 6 nuanced classes (*True → Mostly-True → Half-True → Barely-True → False → Pants-on-Fire*)
- Why **6-class classification is hard**: the boundary between "Mostly-True" and "Half-True" is subtle and subjective
- What **Macro-F1** means and why it's preferred over accuracy for imbalanced datasets
- What **class imbalance** looks like and how it affects evaluation
- Why the LIAR dataset also provides speaker metadata (party, job, history) — and why TRUST Agents deliberately ignores it (to stay evidence-grounded)

### 🔨 Build
- [x] Write `data/load_liar.py` — load LIAR from HuggingFace `datasets`
- [x] Print class distribution (bar chart or table) to understand imbalance
- [x] Write a `map_label_6_to_3()` helper: collapse 6 classes → 3 (`True`, `False`, `Uncertain`) for TRUST Agent output alignment
- [x] Write `evaluation/metrics.py`:
  - `compute_accuracy(preds, labels)`
  - `compute_macro_f1(preds, labels)`
  - `apply_abstention_map(verdicts, mode="pessimistic"|"optimistic")`
- [x] Save 100 samples as `data/liar/sample.jsonl` for rapid testing in later phases

### ✅ Checkpoint
- You can load the dataset and print 5 random examples with their labels
- You can call `compute_macro_f1` on dummy predictions and get a correct score
- You understand why a model that always predicts "False" doesn't score well on Macro-F1

---

## Phase 2 — Agent 1: Claim Extractor

### 📚 Learn
This phase teaches you two fundamental NLP concepts:

**Named Entity Recognition (NER):**
- NER identifies *who*, *what*, *where*, *when* in a text
- Entities like "Senator Biden", "3.5%", "January 2024" are the anchors that make a statement verifiable
- spaCy runs NER as part of its processing pipeline

**Dependency Parsing:**
- Every sentence has a grammatical structure: subject → verb → object
- Dependency parsing extracts this as a tree: *"Biden (subj) signed (verb) the bill (obj)"*
- Factual claims are almost always anchored to a subject-verb-object triplet

**LLM-based Extraction:**
- NER + parsing still miss implicit or rhetorically embedded claims
- An LLM given a structured prompt can identify *"what is the verifiable assertion here?"* for edge cases

### 🔨 Build
- [x] Write `agents/claim_extractor.py`:
  - `extract_with_spacy_ner(text)` → list of entity-anchored claim candidates
  - `extract_with_dependency_parser(text)` → subject-verb-object triplets using dep. parsing
  - `refine_with_llm(candidates, text)` → Ollama LLM call to deduplicate and rewrite as clean claim strings (with heuristic fallback)
  - `extract_claims(text)` → master function combining all three
- [x] Write `prompts/claim_extractor.txt` — the LLM system + user prompt
- [x] Unit test with real news headlines and LIAR benchmark statements

### ✅ Checkpoint
Given input:
```
"The US unemployment rate fell to 3.5% in January 2024, the lowest since 1969."
```
Output successfully decomposed into discrete verifiable factual claims:
- `"The US unemployment rate fell to 3.5%"`
- `"The US unemployment rate fell in January 2024"`

---

## Phase 3 — Agent 2: Hybrid Retrieval (BM25 + FAISS)

### 📚 Learn
This is the **most important phase in the entire project** — retrieval quality determines everything downstream.

**TF-IDF & BM25 (Sparse Retrieval):**
- BM25 is a ranking function that scores documents by *term frequency* and *inverse document frequency*
- Great for: exact names, dates, specific numbers, uncommon keywords
- Weakness: misses documents that describe the same thing in different words ("fell" vs "dropped")

**Sentence Embeddings & FAISS (Dense Retrieval):**
- A SentenceTransformer model converts text into a vector (a list of ~768 numbers)
- Semantically similar sentences have vectors that are close together in vector space
- FAISS (Facebook AI Similarity Search) finds the nearest vectors at speed
- Great for: paraphrases, synonyms, semantic similarity
- Weakness: can retrieve topically related but factually irrelevant documents

**Hybrid Retrieval:**
- Combine BM25 scores + FAISS scores with a weighted sum
- Gets the best of both: exact match precision + semantic recall

**Knowledge Base:**
- You need a corpus to retrieve from — for the project, this will be a local collection of news articles and Wikipedia excerpts relevant to LIAR statements

### 🔨 Build
- [x] Write `data/build_corpus.py` — load, clean, chunk, and index the 600-document knowledge corpus
- [x] Write `retrieval/bm25_index.py`:
  - `BM25Searcher(documents)` — tokenize and index with Okapi BM25
  - `search(query, top_k)` → ranked doc list with lexical scores
- [x] Write `retrieval/faiss_index.py`:
  - `FAISSSearcher(documents)` — embed with SBERT (`all-MiniLM-L6-v2`), store in FAISS FlatIP index
  - `search(query, top_k)` → nearest neighbors with cosine similarity
- [x] Write `retrieval/hybrid_retriever.py`:
  - `HybridRetriever(bm25_weight=0.4, faiss_weight=0.6)` → min-max normalized weighted fusion
- [x] Write `agents/retrieval_agent.py` — Agent 2 module providing `retrieve_evidence_for_claims`

### ✅ Checkpoint
- Exact match queries (e.g. `"The US unemployment rate was 3.5% in January 2024"`) retrieve exact Bureau of Labor Statistics archival evidence with a top score of 1.0.
- Semantic paraphrase queries (e.g. `"American jobless figures at the start of 2024"`) successfully retrieve job/employment evidence via FAISS dense matching where BM25 would struggle.

---

## Phase 4 — Agent 3: Verifier Agent

### 📚 Learn
This phase teaches you:

**Prompt Engineering for Structured Output:**
- LLMs are instruction-followers, but they need precise prompts to give structured, reliable outputs
- You'll learn how to write prompts that consistently return `verdict + confidence` in a parseable format
- You'll learn about few-shot prompting (giving examples in the prompt) to improve consistency

**Calibrated Confidence:**
- A confidence score of 0.9 should mean "I'm right ~90% of the time when I say this"
- LLMs are often overconfident — understanding this limitation is key
- You'll implement a threshold: below `confidence < 0.5` → output "Uncertain" instead of forcing a verdict

**Evidence-Grounded Reasoning:**
- The verifier must reason *only from retrieved evidence* — not from its training knowledge
- This is enforced by prompt design: *"Using only the passages below, determine if the claim is true or false"*
- This is the key differentiator from a plain LLM call

### 🔨 Build
- [x] Write `graph/state.py` — define `EvidencePassage` and `VerificationResult` dataclasses
- [x] Write `prompts/verifier.txt` — structured prompt with grounded evidence rules and JSON schema
- [x] Write `agents/verifier_agent.py`:
  - `build_verifier_prompt(claim, evidence_passages)` → structured prompt
  - `verify_claim(claim, evidence_passages)` → full verify function returning `VerificationResult`
  - Calibrated thresholding: enforces `Uncertain` whenever confidence < 0.50 or evidence is missing
  - Fallback verifier for zero-downtime execution
- [x] Test across supported, refuted, and unproven claims

### ✅ Checkpoint
- Claim: *"The US unemployment rate was 3.5% in January 2024"* $\rightarrow$ `Verdict: True (Confidence: 85%)`
- Claim: *"The Affordable Care Act mandates free sex change surgeries"* $\rightarrow$ `Verdict: False (Confidence: 82%)`
- Claim: *"Alien spacecraft landed in Central Park last Sunday"* (no evidence) $\rightarrow$ `Verdict: Uncertain (Confidence: 20%)`

---

## Phase 5 — Agent 4: Explainer Agent

### 📚 Learn
This phase teaches you:

**Explainability in AI Systems:**
- An AI system that gives you a verdict without evidence is asking you to trust it blindly
- Explainability = the system shows its work, cites its sources, and lets a human audit every step
- This is what separates TRUST Agents from a BERT classifier: you can *read the reasoning*

**Structured Report Generation:**
- LLMs are excellent at converting structured data (dicts, lists) into readable prose
- You'll learn to pass structured inputs (claim, verdict, evidence list) and prompt the LLM to format a professional fact-check report

**Chain-of-Thought (CoT) Output:**
- The explainer doesn't just say "True" — it walks through: *"The claim states X. Evidence passage 2 from source Y confirms X. Therefore: True."*
- This is Chain-of-Thought reasoning made explicit and human-readable

### 🔨 Build
- [x] Write `prompts/explainer.txt` — instruct for structured report with executive summary, claim breakdown, confidence badges, and citations
- [x] Write `agents/explainer_agent.py`:
  - `build_explainer_prompt(verification_results)` → prompt construction
  - `generate_report(verification_results)` → full multi-claim markdown report
  - `deterministic_report_generator` fallback for instant offline audit formatting
- [x] Test across multiple claims with different verdicts and citations

### ✅ Checkpoint
The generated report outputs structured Markdown with:
- Executive Summary of counts
- Detailed claim-by-claim breakdown with status badges (`[VERIFIED TRUE]`, `[FALSE / REFUTED]`, `[UNCERTAIN]`)
- Explicit blockquote evidence citations with source attribution and retrieval scores
- Methodology note detailing each step in the pipeline

---

## Phase 6 — Baseline LangGraph Pipeline

### 📚 Learn
This phase teaches you **LangGraph** — the orchestration layer that connects all 4 agents.

**What LangGraph Is:**
- LangGraph models your pipeline as a **directed graph** of nodes (agents) and edges (data flow)
- Each node receives a shared `State` object, does its work, and returns updated state
- Edges define routing: which node runs after which, and under what conditions

**Why LangGraph Over Plain Python:**
- Conditional routing: *"if claim is compound → route to Decomposer; else → Explainer"*
- Built-in support for parallelism, retries, streaming, and human-in-the-loop interrupts
- Visualizable graph structure — you can literally render your pipeline as a diagram

**State Management:**
- All agents share one typed dictionary (`TrustAgentState`)
- Agents read from it, write to it — no messy argument passing between functions

### 🔨 Build
- [x] Write `graph/state.py` — full `TrustAgentState` TypedDict definition
- [x] Write `graph/baseline_graph.py`:
  - Define nodes: `claim_extractor_node`, `retrieval_node`, `verifier_node`, `explainer_node`
  - Register linear transitions: `START -> claim_extractor -> retrieval_agent -> verifier_agent -> explainer_agent -> END`
  - Compile pipeline via `StateGraph(TrustAgentState)`
- [x] Write `main.py` — executable entrypoint accepting raw news paragraphs and streaming full execution trace
- [x] End-to-end integration test verifying zero-error execution

### ✅ Checkpoint
End-to-end test with a real news headline:
```
Input:  "The US unemployment rate fell to 3.5% in January 2024, the lowest since 1969. Meanwhile, health care reform legislation was passed to mandate free sex change surgeries."
Output: A complete markdown report with verdict + confidence + cited evidence for each extracted claim
```
The pipeline successfully extracted 3 claims, fetched hybrid evidence, verified claims with calibrated confidence, and generated the audit report.

---

## Phase 7 — Evaluation on LIAR Benchmark

### 📚 Learn
This phase teaches you **rigorous evaluation** — how to know if your system actually works.

**Why Evaluation is Hard for Agentic Systems:**
- TRUST Agents outputs `True / False / Uncertain` but LIAR has 6 labels
- You need to map them: this is the **abstention protocol** from the paper
- **Pessimistic mapping:** Uncertain → False (worst case — if you're not sure, assume wrong)
- **Optimistic mapping:** Uncertain → True (best case — if you're not sure, assume correct)
- Reporting both gives an honest performance range

**What "Macro-F1" Really Means:**
- Accuracy hides performance on rare classes
- Macro-F1 treats every class equally — a model that ignores "Pants-on-Fire" gets punished
- You'll compute both and compare with the paper's reported numbers

### 🔨 Build
- [x] Write `evaluation/run_eval.py`:
  - Iterates over LIAR benchmark test statements
  - Executes full 4-agent LangGraph pipeline per statement
  - Maps verdicts using pessimistic and optimistic abstention protocols
  - Computes Accuracy, Macro-F1, and Abstention Rate
  - Saves structured results to `evaluation/results.json`
- [x] Write `evaluation/report.py` — outputs comparative benchmark summary table

### ✅ Checkpoint
- Evaluation completed across 50 benchmark statements:
  - **Pessimistic Accuracy:** 46.0% (Macro-F1: 0.2100)
  - **Optimistic Accuracy:** 46.0% (Macro-F1: 0.3222)
  - **Abstention Rate:** 86.0%
- Empirical confirmation of the paper's primary finding: **Retrieval is the primary bottleneck**. When external corpus coverage is sparse, the system safely abstains with `Uncertain` rather than fabricating hallucinated verdicts.

---

## Phase 8 — Decomposer Agent (Research Extension)

### 📚 Learn
This phase teaches you:

**LoCal-Style Claim Decomposition:**
- Real news claims are often compound: *"X happened because of Y, leading to Z"* — three verifiable assertions bundled in one sentence
- A monolithic verifier collapses this to one True/False label — losing nuance
- Decomposition splits it into atomic sub-claims, each verifiable independently

**When to Decompose:**
- Not every claim needs decomposition — simple single-assertion claims add latency for no gain
- You'll implement a **compound claim detector**: look for conjunctions ("and", "because", "leading to", "which caused"), clause count, and entity density
- This is a routing decision in LangGraph

### 🔨 Build
- [x] Write `agents/decomposer_agent.py`:
  - `is_compound_claim(claim)` → bool (heuristic detector)
  - `decompose_claim(claim)` → `List[str]` of atomic sub-claims via LLM
- [x] Write `prompts/decomposer.txt`
- [x] Update `graph/extended_graph.py` — add conditional edge from Verifier → Decomposer

### ✅ Checkpoint
Input: `"Biden's infrastructure bill passed in 2021, creating 2 million jobs and reducing carbon emissions by 15%"`

Output should be 3 atomic claims:
1. `"Biden's infrastructure bill passed in 2021"`
2. `"The infrastructure bill created 2 million jobs"`
3. `"The infrastructure bill reduced carbon emissions by 15%"`

---

## Phase 9 — Delphi Multi-Agent Jury

### 📚 Learn
This phase teaches you:

**Multi-Agent Debate and Consensus:**
- A single LLM has a single "perspective" determined by its training
- Multiple agents with different personas reduce individual LLM bias — each lens catches different errors
- This mirrors how academic peer review, judicial panels, and expert committees work

**Persona Prompting:**
- A persona is a system prompt that assigns a role: *"You are a data analyst. You only evaluate numerical and statistical claims..."*
- The same base LLM behaves very differently under different system prompts
- You'll learn how system prompts shape LLM reasoning

**Running LLM Calls in Parallel:**
- Three jury members can evaluate a claim simultaneously — LangGraph supports parallel node execution
- You'll use fan-out edges in LangGraph

### 🔨 Build
- [x] Write `agents/jury_agent.py`:
  - Define 3 persona system prompts: `DATA_ANALYST`, `HISTORIAN`, `POLICY_ANALYST`
  - `run_jury(claim, evidence)` → `List[JuryVerdict]` (one per persona)
- [x] Write `prompts/jury_personas.txt` — three distinct system prompts
- [x] Add jury nodes to `graph/extended_graph.py` as parallel fan-out

### ✅ Checkpoint
Run the jury on a numerical claim and a historical claim. Each persona should produce a different verdict or at least a different confidence score — demonstrating that diverse lenses add value.

---

## Phase 10 — Logic Aggregator

### 📚 Learn
This phase teaches you:

**Neurosymbolic AI:**
- Neural AI (LLMs) is probabilistic and flexible but can be inconsistent
- Symbolic AI (formal logic) is deterministic and consistent but rigid
- Neurosymbolic = use LLMs for soft reasoning, then pipe results into formal logic for hard guarantees
- TRUST Agents is a neurosymbolic system: LLM verdicts → symbolic logic aggregation

**Propositional Logic:**
- AND (∧): compound claim is TRUE only if ALL sub-claims are TRUE
- OR (∨): compound claim is TRUE if ANY sub-claim is TRUE
- NOT (¬): negation — used when the compound claim asserts the opposite
- IMPLIES (→): A → B means "if A is true, then B must be true"

**Why Not Use Another LLM Here?**
- The aggregation step must be deterministic — same inputs always produce same output
- An LLM here would reintroduce non-determinism and hallucination risk
- This is a case where rule-based logic is strictly better than a neural approach

### 🔨 Build
- [x] Write `agents/logic_aggregator.py`:
  - `aggregate_verdicts(atomic_verdicts, operator="AND"|"OR")` → final `VerificationResult`
  - `detect_logical_structure(compound_claim)` → infer the right operator from claim text
  - Include a logic trace in the result (which sub-claim caused the final verdict)
- [x] Wire into `graph/extended_graph.py` — receives jury output, produces aggregated verdict

### ✅ Checkpoint
Input:
```python
atomic_verdicts = [
  ("Biden passed infrastructure bill in 2021", "True", 0.92),
  ("It created 2M jobs", "False", 0.71),
  ("It cut carbon by 15%", "Uncertain", 0.38)
]
operator = "AND"
```
Output: `VerificationResult(verdict="False", confidence=0.71, logic_trace="Sub-claim 2 failed (AND operator)")`

---

## Phase 11 — Full Extended Pipeline & Final Evaluation

### 📚 Learn
This final phase teaches you:

**System Integration:**
- End-to-end integration of all 10 agents/components into one coherent LangGraph
- Routing logic that sends simple claims through the 4-agent baseline and compound claims through the full extension

**Ablation Studies:**
- An ablation study removes one component at a time to measure its contribution
- E.g., "What happens to Macro-F1 if we remove the Delphi jury and use just one verifier?"
- This is how you validate that each component genuinely helps

**Qualitative vs. Quantitative Analysis:**
- Raw F1 numbers tell one story
- Reading actual generated reports tells another — are the citations accurate? Is the reasoning sound?
- A complete evaluation includes both

### 🔨 Build
- [x] Finalize `graph/extended_graph.py` — full pipeline with conditional routing
- [x] Update `evaluation/run_eval.py` — evaluate extended pipeline on LIAR
- [x] Write `evaluation/ablation.py` — run pipeline with/without each extension component
- [x] Write a final `RESULTS.md` comparing:
  - TRUST Baseline (yours) vs. TRUST Extended (yours)
  - vs. BERT / RoBERTa (paper reported numbers)
- [x] Manually review 20 generated reports for qualitative quality

### ✅ Checkpoint
You have:
- Quantitative results in a comparison table
- A working end-to-end demo: paste any news paragraph → get a full fact-check report
- Understanding of *why* retrieval quality was the dominant bottleneck in your runs

---

## Summary Roadmap

```
Phase 0   Environment Setup
Phase 1   LIAR Dataset + Evaluation Metrics
Phase 2   Agent 1 — Claim Extractor
Phase 3   Agent 2 — Hybrid Retrieval (BM25 + FAISS)   ← Most Complex
Phase 4   Agent 3 — Verifier Agent
Phase 5   Agent 4 — Explainer Agent
Phase 6   LangGraph Baseline Pipeline                  ← First Milestone
Phase 7   Baseline Evaluation on LIAR
Phase 8   Decomposer Agent (Extension)
Phase 9   Delphi Multi-Agent Jury (Extension)
Phase 10  Logic Aggregator (Extension)
Phase 11  Full Pipeline + Final Evaluation             ← Finish Line
```

---

## Estimated Time (Self-Paced Learning)

| Phase | Estimated Hours |
|---|---|
| Phase 0 — Setup | 1–2 hrs |
| Phase 1 — LIAR Dataset | 2–3 hrs |
| Phase 2 — Claim Extractor | 3–4 hrs |
| Phase 3 — Hybrid Retrieval | 5–8 hrs |
| Phase 4 — Verifier Agent | 3–4 hrs |
| Phase 5 — Explainer Agent | 2–3 hrs |
| Phase 6 — LangGraph Baseline | 4–5 hrs |
| Phase 7 — Baseline Evaluation | 2–3 hrs |
| Phase 8 — Decomposer | 3–4 hrs |
| Phase 9 — Delphi Jury | 3–4 hrs |
| Phase 10 — Logic Aggregator | 3–4 hrs |
| Phase 11 — Final Pipeline + Eval | 4–6 hrs |
| **Total** | **~35–50 hrs** |
