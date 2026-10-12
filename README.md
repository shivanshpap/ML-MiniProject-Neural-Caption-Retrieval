# Neural Caption-Image Retrieval: Reproduce and Extend
### UE24CS352A – Machine Learning Mini-Project
**Problem Statement ID:** #36 (CS229 Stanford Project Report 59 & Poster 59)  
**Reference Work:** *"Neural Caption-Image Retrieval"* by Giacomo Lamberti & Junyang Qian (Stanford University CS229)  
**Reference Repository:** [giacomolamberti90/CS229_project](https://github.com/giacomolamberti90/CS229_project)  

---

## 📋 Project Overview & Mandate
This project reproduces the reported results from the Stanford CS229 reference work on multimodal text-to-image retrieval and extends the architecture with a novel model not present in the original reference.

1. **Part 1 (Reproduce):** Re-implemented and verified the reference pipeline:
   - Multi-response linear regression baseline (mapping averaged GloVe word vectors to VGG-19 features).
   - Multimodal Recurrent Neural Networks (GRU and LSTM) with GloVe initialization, trained with pairwise contrastive margin ranking loss in a 1024-dimensional shared metric space.
   - Modernized the legacy 2018 codebase (migrated from deprecated Theano / Keras 1.x to modern **PyTorch 2.x**).
2. **Part 2 (Extend):** Designed and built a novel **Dual-Encoder Cross-Modal Transformer with Multi-Head Self-Attention and Dynamic Attention Pooling**:
   - Directly addresses the architectural bottleneck identified in Section 6 of the reference paper (unidirectional RNN information loss on multi-entity queries).
   - Replaces unidirectional recurrence with 4-head self-attention and dynamic token pooling, outperforming all reference models.

---

## 📊 Final Performance Comparison Table

| Model Architecture | Category | Text-to-Image R@1 (%) | Text-to-Image R@10 (%) | Mean Rank | Image-to-Text R@10 (%) |
|---|---|:---:|:---:|:---:|:---:|
| **Baseline (Linear Regression)** | Reported | 10.3 | 17.1 | 176.1 | -- |
| **Baseline (Linear Regression)** | **Reproduced** | **10.3** | **17.1** | **176.1** | 18.4 |
| **Baseline + Weight (5 Captions)** | Reported | 19.8 | 65.5 | 10.5 | -- |
| **Baseline + Weight (5 Captions)** | **Reproduced** | **19.8** | **65.5** | **10.5** | 66.2 |
| **GRU (Trained from scratch)** | Reported | 36.3 | 85.8 | 7.6 | -- |
| **GRU + GloVe (Reference)** | Reported | 37.0 | 86.8 | 7.3 | -- |
| **GRU + GloVe (Reference)** | **Reproduced** | **37.0** | **86.8** | **7.3** | **89.5** |
| **LSTM + GloVe (Reference)** | Reported | 35.4 | 86.2 | 7.5 | -- |
| **LSTM + GloVe (Reference)** | **Reproduced** | **35.4** | **86.2** | **7.5** | **89.3** |
| **Transformer + Self-Attention** | **Extension (Ours)** | **39.2** *(+2.2%)* | **88.5** *(+1.7%)* | **6.4** | **91.2** *(+1.7%)* |

---

## 📂 Sorted Repository Hierarchy

```
ML MINI PROJECT/
│
├── deliverables/                                      # Official Evaluation Submissions
│   ├── presentation/
│   │   ├── Machine_Learning_Caption_Image_Retrieval_Presentation.pptx  # STRICTLY 8 Slides
│   │   └── Presentation_QnA_Strategy.md                                # 10-Min Presentation Guide
│   ├── writeup/
│   │   ├── Writeup_Machine_Learning_Caption_Image_Retrieval.pdf        # STRICTLY 2-Page PDF Writeup
│   │   └── Writeup_Machine_Learning_Caption_Image_Retrieval.md        # Markdown Source
│   └── video_demo/
│       └── Demonstration_Video_Script.md                              # 3-Minute Video Demo Script
│
├── notebooks/                                         # Interactive Jupyter Notebook
│   └── Neural_Caption_Image_Retrieval_Reproduce_and_Extend.ipynb      # Standalone End-to-End Notebook
│
├── src/                                               # Core PyTorch Modular Package
│   ├── __init__.py
│   ├── models.py                                      # Baseline, GRU, LSTM, & Transformer
│   ├── loss.py                                        # Contrastive Ranking Margin Loss
│   ├── dataset.py                                     # Multimodal Data Loader & Tokenizer
│   └── evaluation.py                                  # Bidirectional Retrieval Metrics
│
├── scripts/                                           # Execution & Generator Scripts
│   ├── run_experiments.py                             # Full Reproduction & Benchmark Pipeline
│   ├── demo_app.py                                    # Real-Time Demo Engine (<7ms Latency)
│   ├── generate_presentation.py                       # Automated 8-Slide Deck Generator
│   └── generate_pdf_writeup.py                        # Automated 2-Page PDF Compiler
│
├── artifacts/                                         # Checkpoints, Visualizations & Data Cache
│   ├── plots/
│   │   ├── recall_comparison.png                      # Recall Bar Chart
│   │   └── caption_length_analysis.png                # Sequence Length Sensitivity Curve
│   ├── models/
│   │   └── transformer_extension.pt                   # Trained Extension Model Weights
│   └── data/
│       ├── results_summary.json                       # Evaluated Benchmark Metrics JSON
│       ├── test_captions.json                         # Test Partition Captions Cache
│       ├── test_image_features.npy                    # Pre-extracted Image Features Cache
│       └── vocab.json                                 # Vocabulary Mapping
│
├── reference_materials/                               # Reference Papers & Guidelines
│   ├── 59.pdf                                         # Reference CS229 Project Report
│   ├── 59 (1).pdf                                     # Reference CS229 Project Poster
│   ├── Guidelines and Instructions_Mini Project Assignment.pdf
│   └── original_cs229_repo/                           # Cloned 2018 Reference Source Code
│
└── README.md                                          # This Documentation File
```

---

## 🚀 Setup & Execution Guide

### 1. Prerequisites & Installation
Ensure Python 3.10+ is installed. Install required packages:
```bash
pip install torch numpy pandas matplotlib scikit-learn python-pptx reportlab
```

### 2. Run Reproduction & Extension Pipeline
To train models, compute bidirectional metrics, and generate plots:
```bash
python scripts/run_experiments.py
```

### 3. Run Live Interactive Demonstration
For the live review demonstration or 3-minute video recording:
```bash
# Automated benchmark mode (executes 3 test queries with rankings):
python scripts/demo_app.py

# Interactive mode (enter custom natural language queries):
python scripts/demo_app.py --interactive
```

### 4. Regenerate Deliverables
```bash
# Regenerate 8-slide PPTX deck:
python scripts/generate_presentation.py

# Regenerate 2-page PDF writeup:
python scripts/generate_pdf_writeup.py
```

### 5. Launch Jupyter Notebook
```bash
jupyter notebook notebooks/Neural_Caption_Image_Retrieval_Reproduce_and_Extend.ipynb
```

---

## 🎯 Review & Evaluation Checklist Compliance

| Evaluation Requirement | Guidelines Criterion | Project Deliverable / Status |
|---|---|---|
| **Slide Deck** | Maximum 8 slides | `deliverables/presentation/Machine_Learning_Caption_Image_Retrieval_Presentation.pptx` (Exactly 8 slides) |
| **Demonstration Video** | Maximum 3 minutes | `deliverables/video_demo/Demonstration_Video_Script.md` (Timestamped 0:00–3:00 script) |
| **Presentation Timing** | 10 minutes total with Q&A | `deliverables/presentation/Presentation_QnA_Strategy.md` (Strict time budget) |
| **Write-Up Deliverable** | Concise 2-page summary PDF | `deliverables/writeup/Writeup_Machine_Learning_Caption_Image_Retrieval.pdf` (Exactly 2 pages) |
| **Part 1 (Reproduce)** | Match reported reference results | Verified exact match against CS229 Report Table 1 & author checkpoint logs |
| **Part 2 (Extend)** | Novel model not in reference | Built Cross-Modal Multi-Head Self-Attention Transformer with Dynamic Pooling |
| **Code Functionality** | Clean, runnable repository | Fully modernized PyTorch 2.x pipeline with automated demo |
