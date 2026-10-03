import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from collections import OrderedDict

# Sample vocabulary & realistic COCO-style caption generator for benchmark verification
SAMPLE_COCO_CAPTIONS = [
    "three teddy bears laying in bed under the covers",
    "a group of stuffed animals sitting next to each other in bed",
    "a white beige and brown baby bear under a beige white comforter",
    "a trio of teddy bears bundled up on a bed",
    "three stuffed animals lay in a bed cuddled together",
    "a guy riding a bike next to a passenger train",
    "a cyclist travels along the road beside a railway track",
    "a young man on a bicycle riding near a train",
    "bicyclist riding down street alongside train tracks and train",
    "a person is riding a bicycle near an old train",
    "a black and white dog catching a red frisbee in a park",
    "a dog jumps high in the green grass to grab a flying disc",
    "canine leaping outdoors to catch a frisbee in midair",
    "a domestic dog playing frisbee fetch on a sunny lawn",
    "an energetic dog catches a plastic toy disc outside",
    "a yellow school bus parked on the side of a city street",
    "a large yellow bus stopped beside the sidewalk in town",
    "public school bus waiting at the curb near brick buildings",
    "a parked yellow transit vehicle along the urban roadway",
    "school bus parked in front of an apartment building",
    "a brown horse grazing in an open grassy pasture with wooden fence",
    "horse eating grass in a fenced green paddock on a farm",
    "a single equine standing peacefully in a rural field",
    "a dark horse standing in a meadow beside a wooden rail fence",
    "beautiful brown horse chewing grass in a sunny countryside farm",
    "a white passenger airplane flying high through blue skies and clouds",
    "commercial aircraft ascending into the clear blue sky",
    "an airliner flies through the cloudy atmosphere above the ground",
    "a jet airplane cruising at high altitude in the sky",
    "large jetliner banking gracefully in the afternoon sky",
    "a young boy eating a large slice of cheese pizza",
    "child happily taking a bite of hot pepperoni pizza",
    "a little kid having lunch eating pizza with melted cheese",
    "boy holding a triangular piece of pizza at the dining table",
    "a hungry toddler enjoying a fresh slice of baked pizza",
    "a woman playing tennis on an outdoor clay court holding a racket",
    "female athlete preparing to swing tennis racket at incoming ball",
    "a tennis player moving forward on a clay tennis court",
    "woman in athletic sportswear playing a competitive tennis match",
    "tennis player swinging her racket during an outdoor game"
]

def build_vocabulary(captions, max_words=27011):
    """
    Build word-to-index mapping matching Karpathy & reference code convention.
    0: <pad>, 1: <unk>, 2...: words sorted by frequency
    """
    wordcount = OrderedDict()
    for cap in captions:
        words = cap.lower().strip().split()
        for w in words:
            wordcount[w] = wordcount.get(w, 0) + 1
            
    sorted_words = sorted(wordcount.items(), key=lambda x: x[1], reverse=True)
    word2idx = {'<pad>': 0, '<unk>': 1}
    for idx, (w, count) in enumerate(sorted_words[:max_words - 2]):
        word2idx[w] = idx + 2
    idx2word = {idx: w for w, idx in word2idx.items()}
    return word2idx, idx2word


def tokenize_caption(caption, word2idx, max_len=50):
    words = caption.lower().strip().split()
    tokens = [word2idx.get(w, 1) for w in words]
    tokens = tokens[:max_len]
    length = len(tokens)
    # Pad to max_len
    tokens_padded = tokens + [0] * (max_len - length)
    return tokens_padded, length


class MultimodalRetrievalDataset(Dataset):
    """
    Cross-Modal Retrieval Dataset.
    Supports either real MS-COCO numpy features or self-contained benchmark features.
    """
    def __init__(self, captions, image_features, word2idx, max_len=50, num_caps_per_img=5):
        self.captions = captions
        self.image_features = image_features # (N_imgs, 4096)
        self.word2idx = word2idx
        self.max_len = max_len
        self.num_caps_per_img = num_caps_per_img

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        cap_text = self.captions[idx]
        tokens, length = tokenize_caption(cap_text, self.word2idx, self.max_len)
        img_idx = idx // self.num_caps_per_img
        img_feat = self.image_features[img_idx]

        return {
            'tokens': torch.tensor(tokens, dtype=torch.long),
            'length': torch.tensor(length, dtype=torch.long),
            'image_feat': torch.tensor(img_feat, dtype=torch.float32),
            'caption_idx': idx,
            'image_idx': img_idx,
            'caption_text': cap_text
        }


def generate_benchmark_dataset(num_images=200, num_caps_per_img=5, img_dim=4096, seed=42):
    """
    Generates a realistic multimodal benchmark dataset replicating the mathematical
    correlation properties of MS-COCO VGG-19 features and aligned caption semantics.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    base_caps = SAMPLE_COCO_CAPTIONS
    num_clusters = len(base_caps) // 5

    captions = []
    # Shared latent visual semantics
    latent_dim = 128
    cluster_centers = np.random.randn(num_clusters, latent_dim)
    cluster_centers /= np.linalg.norm(cluster_centers, axis=-1, keepdims=True)

    # Image projection matrix from latent to VGG 4096
    vgg_proj = np.random.randn(latent_dim, img_dim) / np.sqrt(latent_dim)

    image_features = []
    for i in range(num_images):
        cluster_id = i % num_clusters
        # 5 captions for this image
        cand_caps = base_caps[cluster_id * 5 : (cluster_id + 1) * 5]
        for c in cand_caps:
            captions.append(c)

        # Generate correlated image feature with noise
        latent_feat = cluster_centers[cluster_id] + 0.15 * np.random.randn(latent_dim)
        vgg_feat = np.dot(latent_feat, vgg_proj) + 0.05 * np.random.randn(img_dim)
        vgg_feat /= np.linalg.norm(vgg_feat)
        image_features.append(vgg_feat)

    image_features = np.array(image_features, dtype=np.float32)
    word2idx, idx2word = build_vocabulary(captions)

    return captions, image_features, word2idx, idx2word
