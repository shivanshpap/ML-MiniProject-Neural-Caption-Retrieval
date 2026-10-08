import os
import sys
import json
import torch
import numpy as np
import time

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.dataset import tokenize_caption, SAMPLE_COCO_CAPTIONS
from src.models import SelfAttentionTransformerEncoder, GRUWithGloVe

def run_demo(interactive=False):
    print("\n" + "=" * 75)
    print("      NEURAL CAPTION-IMAGE RETRIEVAL: LIVE DEMONSTRATION SYSTEM")
    print("      CS229 Stanford Project Reproduction & Self-Attention Extension")
    print("=" * 75)

    # Load cache and artifacts from artifacts/ folder
    vocab_file = os.path.join(PROJECT_ROOT, 'artifacts', 'data', 'vocab.json')
    captions_file = os.path.join(PROJECT_ROOT, 'artifacts', 'data', 'test_captions.json')
    images_file = os.path.join(PROJECT_ROOT, 'artifacts', 'data', 'test_image_features.npy')
    transformer_weights = os.path.join(PROJECT_ROOT, 'artifacts', 'models', 'transformer_extension.pt')

    if not (os.path.exists(vocab_file) and os.path.exists(images_file)):
        print("\n[!] Precomputed benchmark files not found. Running run_experiments.py first...")
        from scripts.run_experiments import run_reproduce_and_extend
        run_reproduce_and_extend()

    with open(vocab_file, 'r') as f:
        word2idx = json.load(f)
    with open(captions_file, 'r') as f:
        test_caps = json.load(f)
    test_imgs = np.load(images_file)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vocab_size = len(word2idx)
    num_test_images = len(test_imgs)

    print(f"\n[+] Loaded Test Image Database: {num_test_images} images (4096-dim VGG-19 features)")
    print(f"[+] Loaded Vocabulary: {vocab_size} tokens")

    # Load Extension Model
    print("[+] Loading Cross-Modal Transformer Encoder (Extension Model)...")
    trans_model = SelfAttentionTransformerEncoder(
        vocab_size=vocab_size, word_dim=300, d_model=256, nhead=4, num_layers=2, shared_dim=256, image_dim=4096
    ).to(device)
    if os.path.exists(transformer_weights):
        trans_model.load_state_dict(torch.load(transformer_weights, map_location=device))
    trans_model.eval()

    # Precompute image embeddings
    test_imgs_tensor = torch.tensor(test_imgs, dtype=torch.float32).to(device)
    with torch.no_grad():
        image_embeddings = trans_model.encode_image(test_imgs_tensor).cpu().numpy()

    # Demo queries
    demo_queries = [
        "a guy riding a bike next to a passenger train",
        "three teddy bears laying in bed under the covers",
        "a black and white dog catching a red frisbee in a park",
        "a yellow school bus parked on the side of a city street",
        "a white passenger airplane flying high through blue skies and clouds"
    ]

    def search_query(query_text):
        print("\n" + "-" * 75)
        print(f"QUERY CAPTION: \"{query_text}\"")
        print("-" * 75)
        tokens, length = tokenize_caption(query_text, word2idx)
        tok_tensor = torch.tensor([tokens], dtype=torch.long).to(device)

        start_time = time.time()
        with torch.no_grad():
            cap_emb = trans_model.encode_text(tok_tensor).cpu().numpy() # (1, 256)
            diff = cap_emb - image_embeddings # (30, 256)
            distances = np.sum(diff ** 2, axis=-1) # (30,)
            sorted_indices = np.argsort(distances)
        query_time = (time.time() - start_time) * 1000

        print(f"Nearest Neighbor Retrieval Time: {query_time:.2f} ms")
        print("\nTOP-5 RETRIEVED IMAGE CANDIDATES:")
        print(f"{'Rank':<6} | {'Image ID':<10} | {'Distance Score':<16} | {'Associated Ground Truth Caption'}")
        print("-" * 75)

        for rank, img_idx in enumerate(sorted_indices[:5], 1):
            sample_cap = test_caps[img_idx * 5] if (img_idx * 5 < len(test_caps)) else "COCO Image"
            print(f"#{rank:<5} | Image_{img_idx:03d}  | {distances[img_idx]:<16.4f} | \"{sample_cap}\"")

        # Ground truth check
        matched = False
        for pos, img_idx in enumerate(sorted_indices):
            for c_idx in range(img_idx * 5, min((img_idx + 1) * 5, len(test_caps))):
                if test_caps[c_idx].lower() == query_text.lower():
                    print(f"\n>> Ground Truth Match Found at Rank: #{pos + 1} (Recall@1: {pos == 0}, Recall@10: {pos < 10})")
                    matched = True
                    break
            if matched:
                break

    if not interactive:
        print("\n>>> RUNNING AUTOMATED DEMO BENCHMARK (Ideal for 3-minute video presentation) <<<")
        for q in demo_queries[:3]:
            search_query(q)
            time.sleep(0.5)
        print("\n" + "=" * 75)
        print("Demo completed successfully. For interactive mode run: python demo_app.py --interactive")
        print("=" * 75)
    else:
        print("\n>>> INTERACTIVE DEMONSTRATION MODE <<<")
        print("Type a natural language query or choose from sample queries. Type 'exit' to quit.\n")
        while True:
            try:
                user_input = input("Enter caption query > ").strip()
                if not user_input or user_input.lower() == 'exit':
                    break
                search_query(user_input)
            except (KeyboardInterrupt, EOFError):
                break

if __name__ == '__main__':
    is_interactive = '--interactive' in sys.argv
    run_demo(interactive=is_interactive)
