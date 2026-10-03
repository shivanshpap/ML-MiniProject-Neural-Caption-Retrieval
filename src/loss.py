import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveRankingLoss(nn.Module):
    """
    Pairwise Contrastive Ranking Loss from CS229 Project (Equation 2):
    Cost = sum_{(c,i)} [ sum_{c'} max(0, alpha - S(c,i) + S(c',i)) + sum_{i'} max(0, alpha - S(c,i) + S(c,i')) ]
    where S(c,i) = - || f_i(i) - f_c(c) ||_2^2 or cosine similarity.
    """
    def __init__(self, margin=0.1, similarity_type='euclidean'):
        super().__init__()
        self.margin = margin
        self.similarity_type = similarity_type

    def forward(self, caption_emb, image_emb):
        """
        caption_emb: (batch_size, embed_dim)
        image_emb:   (batch_size, embed_dim)
        """
        batch_size = caption_emb.size(0)

        if self.similarity_type == 'euclidean':
            # Pairwise squared euclidean distances: (B, B)
            diff = caption_emb.unsqueeze(1) - image_emb.unsqueeze(0) # (B, B, D)
            errors = torch.sum(diff ** 2, dim=-1) # (B, B)
            # S(c, i) = - errors
            diagonal = torch.diag(errors) # S(c_k, i_k) positive pairs

            # Cost over contrastive images: max(0, margin - errors_contrastive + error_diagonal)
            # errors is (c, i) error
            cost_im = torch.clamp(self.margin - errors + diagonal.unsqueeze(1), min=0.0)
            # Cost over contrastive captions
            cost_s = torch.clamp(self.margin - errors + diagonal.unsqueeze(0), min=0.0)

            # Zero out diagonal
            mask = torch.eye(batch_size, dtype=torch.bool, device=caption_emb.device)
            cost_im.masked_fill_(mask, 0.0)
            cost_s.masked_fill_(mask, 0.0)

            loss = (cost_im + cost_s).sum() / batch_size
            return loss

        elif self.similarity_type == 'cosine':
            caption_norm = F.normalize(caption_emb, p=2, dim=-1)
            image_norm = F.normalize(image_emb, p=2, dim=-1)
            sim_matrix = torch.matmul(caption_norm, image_norm.t()) # (B, B)
            diagonal = torch.diag(sim_matrix)

            cost_im = torch.clamp(self.margin - diagonal.unsqueeze(1) + sim_matrix, min=0.0)
            cost_s = torch.clamp(self.margin - diagonal.unsqueeze(0) + sim_matrix, min=0.0)

            mask = torch.eye(batch_size, dtype=torch.bool, device=caption_emb.device)
            cost_im.masked_fill_(mask, 0.0)
            cost_s.masked_fill_(mask, 0.0)

            loss = (cost_im + cost_s).sum() / batch_size
            return loss
