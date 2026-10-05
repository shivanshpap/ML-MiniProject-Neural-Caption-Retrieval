import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LinearBaseline(nn.Module):
    """
    Baseline Model (Section 4.1):
    Multi-response linear regression mapping averaged GloVe word vectors to image features.
    argmin_W sum || f_VGG(i_k) - W * f_GloVe(c_k) ||^2
    """
    def __init__(self, text_dim=300, image_dim=4096, shared_dim=1024):
        super().__init__()
        self.text_proj = nn.Linear(text_dim, shared_dim, bias=False)
        self.image_proj = nn.Linear(image_dim, shared_dim, bias=False)

    def forward(self, text_features, image_features):
        """
        text_features: (B, 300) - GloVe averaged sentence vectors
        image_features: (B, 4096) - VGG-19 features
        """
        t_emb = self.text_proj(text_features)
        i_emb = self.image_proj(image_features)
        return t_emb, i_emb


class GRUWithGloVe(nn.Module):
    """
    Multimodal Neural Network with GRU (Section 4.2):
    - GloVe word embeddings (300-d)
    - Unidirectional GRU language model (hidden_dim = 1024)
    - Linear image projection: W_i * f_VGG (4096 -> 1024)
    """
    def __init__(self, vocab_size=27011, word_dim=300, hidden_dim=1024, image_dim=4096):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, word_dim, padding_idx=0)
        self.gru = nn.GRU(word_dim, hidden_dim, batch_first=True)
        self.image_proj = nn.Linear(image_dim, hidden_dim, bias=True)

    def encode_text(self, token_ids, lengths=None):
        # token_ids: (B, max_len)
        x = self.embedding(token_ids) # (B, L, 300)
        out, h_n = self.gru(x)        # h_n: (1, B, 1024)
        c_emb = h_n.squeeze(0)        # (B, 1024)
        return c_emb

    def encode_image(self, image_features):
        # image_features: (B, 4096)
        return self.image_proj(image_features)

    def forward(self, token_ids, image_features, lengths=None):
        c_emb = self.encode_text(token_ids, lengths)
        i_emb = self.encode_image(image_features)
        return c_emb, i_emb


class LSTMWithGloVe(nn.Module):
    """
    Multimodal Neural Network with LSTM (Section 4.2 / Table 1):
    - GloVe word embeddings (300-d)
    - LSTM language model (hidden_dim = 1024)
    - Linear image projection: W_i * f_VGG (4096 -> 1024)
    """
    def __init__(self, vocab_size=27011, word_dim=300, hidden_dim=1024, image_dim=4096):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, word_dim, padding_idx=0)
        self.lstm = nn.LSTM(word_dim, hidden_dim, batch_first=True)
        self.image_proj = nn.Linear(image_dim, hidden_dim, bias=True)

    def encode_text(self, token_ids, lengths=None):
        x = self.embedding(token_ids)
        out, (h_n, c_n) = self.lstm(x)
        return h_n.squeeze(0)

    def encode_image(self, image_features):
        return self.image_proj(image_features)

    def forward(self, token_ids, image_features, lengths=None):
        c_emb = self.encode_text(token_ids, lengths)
        i_emb = self.encode_image(image_features)
        return c_emb, i_emb


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # x: (B, L, D)
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class AttentionPooling(nn.Module):
    """
    Dynamic Attention Pooling over sequence tokens:
    alpha_t = softmax(w^T tanh(W_a h_t))
    c_emb = sum_t alpha_t * h_t
    """
    def __init__(self, d_model):
        super().__init__()
        self.attn_net = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1, bias=False)
        )

    def forward(self, sequence_hidden, mask=None):
        # sequence_hidden: (B, L, D)
        scores = self.attn_net(sequence_hidden).squeeze(-1) # (B, L)
        if mask is not None:
            scores = scores.masked_fill(~mask, -1e9)
        weights = F.softmax(scores, dim=-1).unsqueeze(-1) # (B, L, 1)
        pooled = torch.sum(sequence_hidden * weights, dim=1) # (B, D)
        return pooled, weights


class SelfAttentionTransformerEncoder(nn.Module):
    """
    EXTENSION MODEL:
    Dual-Encoder Cross-Modal Transformer with Multi-Head Self-Attention
    and Dynamic Attention Pooling.
    
    Addresses the limitation stated in Paper Section 6:
    Unidirectional RNNs suffer from vanishing gradients and cannot dynamically weigh
    multi-object interactions (e.g. 'a guy riding a bike next to a train').
    Multi-Head Self-Attention enables all token pairs to attend to each other,
    and Dynamic Attention Pooling extracts the visual-salient semantic anchors.
    """
    def __init__(self, vocab_size=27011, word_dim=300, d_model=512, nhead=4,
                 num_layers=2, shared_dim=1024, image_dim=4096, dropout=0.1):
        super().__init__()
        # Text Encoder
        self.embedding = nn.Embedding(vocab_size, word_dim, padding_idx=0)
        self.input_proj = nn.Linear(word_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.attn_pool = AttentionPooling(d_model)
        self.text_head = nn.Sequential(
            nn.Linear(d_model, shared_dim),
            nn.LayerNorm(shared_dim)
        )

        # Image Encoder: Deep Non-Linear Projection Network
        self.image_head = nn.Sequential(
            nn.Linear(image_dim, 2048),
            nn.LayerNorm(2048),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2048, shared_dim),
            nn.LayerNorm(shared_dim)
        )

    def encode_text(self, token_ids, mask=None):
        # token_ids: (B, L)
        padding_mask = (token_ids != 0) if mask is None else mask
        x = self.embedding(token_ids)
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        h = self.transformer(x, src_key_padding_mask=~padding_mask)
        pooled, weights = self.attn_pool(h, padding_mask)
        c_emb = self.text_head(pooled)
        return c_emb

    def encode_image(self, image_features):
        return self.image_head(image_features)

    def forward(self, token_ids, image_features, mask=None):
        c_emb = self.encode_text(token_ids, mask)
        i_emb = self.encode_image(image_features)
        return c_emb, i_emb
