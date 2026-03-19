# diffusiontransformer-indian-language-systems
# OKM Conditional Diffusion Transformer

End-to-end pipeline that takes an English sentence and generates an 
**Object Knowledge Model (OKM)** knowledge graph, as defined in the paper:
*"A Novel Approach to Extend KM Models with OKM and Kafka for Big Data and Semantic Web"*.

---

## Architecture Overview

```
Sentence
   │
   ▼
┌─────────────────────────────────────┐
│         GRAPH GENERATOR             │
│                                     │
│  Stage 1: Linguistic Preprocessor   │  ← spaCy neural model (pre-trained)
│           ↓                         │
│  Stage 2: OKM Attribute Assigner    │  ← Pure rules (dep label → OKM case)
│           ↓                         │
│  Stage 3: Ambiguity Resolver        │  ← Small BERT classifier (optional)
│           ↓                         │
│  Stage 4: Graph Constructor         │  ← Pure rules → nx.DiGraph
└──────────────────┬──────────────────┘
                   │ (sentence, OKM graph) pairs
                   ▼
┌─────────────────────────────────────┐
│      CONDITIONAL DIFFUSION MODEL    │
│                                     │
│  Graph → Tensors (X, E)             │
│       + Noise Schedule              │
│       + Sentence Encoder (BERT)     │
│       + Graph Transformer Denoiser  │
│       → Output Heads                │
└─────────────────────────────────────┘
```

---

## Project Structure

```
okm_pipeline/
│
├── graph_generator/
│   ├── stage1_linguistic.py       # spaCy POS + dependency parsing
│   ├── stage2_attribute_assigner.py  # OKM case/tense/gender rules
│   ├── stage3_ambiguity_resolver.py  # BERT-based ambiguity resolution
│   ├── stage4_graph_constructor.py   # NetworkX graph assembly
│   └── pipeline.py                # Combines all 4 stages
│
├── diffusion/
│   ├── graph_representation.py    # Graph ↔ tensor conversion
│   ├── noise_schedule.py          # DDPM/cosine noise schedule
│   ├── model.py                   # Graph Transformer denoiser
│   ├── train.py                   # Training loop
│   └── inference.py               # DDIM/DDPM sampling + CFG
│
├── data/
│   └── dataset.py                 # Dataset class + DataLoaders
│
├── main.py                        # CLI entry point
└── requirements.txt
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Run the demo (no training needed)
```bash
python main.py demo
```

### 3. Generate training data from your own sentences
```bash
# Put one sentence per line in data/sentences.txt
python main.py generate_data --sentences_file data/sentences.txt
```

### 4. Train the model
```bash
python main.py train --epochs 100 --batch_size 16
```

### 5. Run inference
```bash
python main.py infer --sentence "She went to the market by bus."
```

---

## OKM Notation

Each word gets attributed according to OKM grammar from the paper:

| Word type | Notation       | Attributes                          |
|-----------|----------------|-------------------------------------|
| Noun      | `Ram(1,1,1)`   | `(gender, number, case)`            |
| Verb      | `went(3,1,-1)` | `(person, number, tense)`           |
| Descriptor| `[d:the]`      | —                                   |

**Cases (from Sanskrit grammar):**
1. Subjective, 2. Objective, 3. Instrumental, 4. Dative,
5. Ablative, 6. Genitive, 7. Locative, 8. Vocative

**Tense:** -1 = past, 0 = present, 1 = future

---

## Training Details

- **Diffusion:** Cosine noise schedule, 1000 timesteps
- **Sampling:** DDIM (50 steps) for fast inference
- **Conditioning:** Sentence-BERT embedding + Classifier-Free Guidance
- **Losses:** Node/Edge noise MSE + Node type CE + Edge BCE + Word CE
- **Model size:** ~15M parameters (default config)

---

## Extending to Other Languages

As described in the paper, translate non-English sentences to English first
(e.g. using a translation library), then run through the pipeline.
The OKM grammar is language-agnostic once the sentence is in English.


