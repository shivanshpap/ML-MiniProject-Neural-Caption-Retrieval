# 3-Minute Demonstration Video Script & Recording Guide
## Project: Neural Caption-Image Retrieval (Reproduce & Extend)
**Course:** UE24CS352A – Machine Learning Mini-Project  
**Topic:** Neural Caption-Image Retrieval (CS229 Reference #36)  
**Maximum Allowed Video Duration:** 3 Minutes (180 Seconds)  

---

### Overview & Video Recording Checklist
- **Tool to Record:** OBS Studio, Loom, Zoom, or Windows Game Bar (`Win + Alt + R`).
- **Screen Layout:** Terminal / Code Editor on one side, output / visual charts on the other.
- **Microphone:** Clear voice, steady pace, follow the time-stamped cues below.
- **Command to Run During Demo:** `python demo_app.py`

---

### Timestamped Video Script (0:00 – 3:00)

| Timestamp | Video Screen Action | Spoken Script / Narration |
|---|---|---|
| **0:00 – 0:25** (25s) | **Screen:** Show Slide 1 or Title Slide of Presentation / Codebase root.<br>**Visual:** Display repository layout: `src/`, `output/`, `demo_app.py`. | *"Hello everyone. Welcome to the demonstration of our Machine Learning Mini-Project on **Neural Caption-Image Retrieval**, based on the Stanford CS229 reference work by Giacomo Lamberti and Junyang Qian. Our project addresses cross-modal retrieval on the Microsoft COCO dataset: retrieving relevant images given a natural language descriptive query in a joint 1024-dimensional semantic embedding space."* |
| **0:25 – 0:55** (30s) | **Screen:** Switch to Terminal. Show `run_experiments.py` results table and `output/recall_comparison.png`. | *"In **Part 1**, we reproduced the reference paper's core models: the multi-response linear regression baseline and the deep recurrent models with GRU and LSTM initialized with 300-dimensional GloVe word vectors. As shown in our evaluation benchmark, our reproduced GRU+GloVe model achieves **37.0% Recall@1** and **86.8% Recall@10**, aligning with the author's reported results in Table 1."* |
| **0:55 – 1:30** (35s) | **Screen:** Open `src/models.py` showing `SelfAttentionTransformerEncoder`. Show `output/caption_length_analysis.png`. | *"In **Part 2**, we extended the work by addressing the core limitation noted in Section 6 of the paper: unidirectional RNNs struggle with long captions and multi-entity binding. We built an additional model: a **Dual-Encoder Cross-Modal Transformer with Multi-Head Self-Attention and Dynamic Attention Pooling**. This model allows every word to attend to all others simultaneously. As shown in our comparative analysis, it surpasses the reference models, achieving **39.2% Recall@1** (+2.2% gain), **88.5% Recall@10**, and reduces the Mean Rank to **6.4**."* |
| **1:30 – 2:30** (60s) | **Screen:** Terminal window. Run command: `python demo_app.py`<br>**Visual:** Show live execution, prompt queries, and sub-7ms retrieval latency. | *"Now let's watch the working system in real-time. I will run `python demo_app.py`.*<br><br>*(Points to Terminal)* *Here the system loads our trained Cross-Modal Transformer and 4096-dimensional VGG-19 image features.*<br><br>*Test Query 1: 'a guy riding a bike next to a passenger train'. Within just 5.8 milliseconds, the system ranks all candidate images and successfully places the true ground truth image at **Rank #1** with minimum distance score!*<br><br>*Test Query 2: 'three teddy bears laying in bed under the covers'. Again, the model retrieves the exact target image at **Rank #1**!*<br><br>*Notice how the Self-Attention mechanism correctly parses complex sentences with multiple interacting entities where the baseline linear model previously collapsed."* |
| **2:30 – 3:00** (30s) | **Screen:** Return to Slide 8 or Master Summary Table. | *"In conclusion, we successfully reproduced the reported CS229 baseline and RNN results and demonstrated clear quantitative gains through our proposed Multi-Head Self-Attention extension. The complete pipeline is implemented in PyTorch with reproducible training and evaluation scripts. Thank you!"* |

---

### Tips for Perfect Video Demonstration
1. **Practice Once:** Run `python demo_app.py` once beforehand so weights and cache are loaded.
2. **Audio Clarity:** Speak with confidence; do not rush, but do not pause excessively.
3. **Strict Time Limit:** Ensure your final recorded video length is between **2:40 and 2:55 minutes** so you never exceed the 3-minute penalty limit.
