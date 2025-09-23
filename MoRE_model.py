# ---------- MoRE: Enhanced Multi-Omics Representation Embedding ----------
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class SupConLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        features = F.normalize(features, dim=1)
        sim_matrix = torch.matmul(features, features.T) / self.temperature
        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.T).float()
        logits_mask = torch.ones_like(mask) - torch.eye(mask.shape[0], device=mask.device)
        mask *= logits_mask
        exp_sim = torch.exp(sim_matrix) * logits_mask
        log_prob = sim_matrix - torch.log(exp_sim.sum(1, keepdim=True) + 1e-12)
        mean_log_prob = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-12)
        return -mean_log_prob.mean()

class FrozenTransformerBackbone(nn.Module):
    def __init__(self, input_dim, embed_dim, nhead=4, nlayers=2):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, batch_first=True),
            num_layers=nlayers
        )
        for p in self.encoder.parameters():
            p.requires_grad = False

    def forward(self, x):
        x = self.proj(x).unsqueeze(1)
        x = self.encoder(x).squeeze(1)
        return x

class TaskAdaptiveFusion(nn.Module):
    def __init__(self, embed_dim, num_modalities):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(num_modalities, embed_dim))

    def forward(self, modality_embeddings):
        stack = torch.stack(modality_embeddings)
        return (stack * self.weights[:, None, :]).sum(dim=0)

class BatchCorrection(nn.Module):
    def __init__(self, embed_dim, num_batches):
        super().__init__()
        self.batch_embed = nn.Embedding(num_batches, embed_dim)

    def forward(self, x, batch_labels):
        batch_effect = self.batch_embed(batch_labels)
        return x - batch_effect

class IterativeRefinement(nn.Module):
    def __init__(self, embed_dim, steps=3):
        super().__init__()
        self.steps = steps
        self.refine = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, x):
        for _ in range(self.steps):
            x = x + self.refine(x)
        return x

def intra_class_variance_loss(latent, labels):
    loss = 0.0
    for c in labels.unique():
        idx = (labels == c).nonzero(as_tuple=True)[0]
        if len(idx) < 2: continue
        cluster = latent[idx]
        center = cluster.mean(dim=0, keepdim=True)
        loss += ((cluster - center) ** 2).mean()
    return loss

class MoRE(nn.Module):
    def __init__(self, input_dims, embed_dim, num_classes, n_batches, nhead=4, nlayers=2, refine_steps=3):
        super().__init__()
        self.backbones = nn.ModuleList([
            FrozenTransformerBackbone(dim, embed_dim, nhead, nlayers) for dim in input_dims
        ])
        self.fusion = TaskAdaptiveFusion(embed_dim, len(input_dims))
        self.batch_correction = BatchCorrection(embed_dim, n_batches)
        self.refiner = IterativeRefinement(embed_dim, steps=refine_steps)
        self.classifier = nn.Linear(embed_dim, num_classes)

        self.supcon_loss = SupConLoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.align_loss = nn.MSELoss()

    def forward(self, inputs, labels=None, batch_labels=None):
        modality_embeddings = [backbone(x) for backbone, x in zip(self.backbones, inputs)]
        fused = self.fusion(modality_embeddings)
        if batch_labels is not None:
            fused = self.batch_correction(fused, batch_labels)

        refined = self.refiner(fused)
        logits = self.classifier(refined)

        losses = {}
        if labels is not None:
            losses['ce'] = self.ce_loss(logits, labels)
            losses['supcon'] = self.supcon_loss(refined, labels)
            losses['align'] = self.align_loss(modality_embeddings[0], modality_embeddings[1])
            losses['intra_var'] = intra_class_variance_loss(refined, labels)
            losses['total'] = (
                1.0 * losses['ce'] +
                1.5 * losses['supcon'] +
                0.5 * losses['align'] +
                1.0 * losses['intra_var']
            )

        return logits, refined, modality_embeddings, losses
