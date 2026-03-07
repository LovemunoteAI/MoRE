# MoRE

**MoRE (Multi-Omics Representation Embedding)** is an LLM-inspired framework that repurposes frozen pre-trained transformer backbones for multi-omics integration.

---

## Overview

Representation learning on multi-omics data is challenging due to:

- extreme dimensionality  
- modality heterogeneity  
- cohort-specific batch effects  

While pre-trained transformer backbones have shown broad generalization capabilities in biological sequence modeling, their application to multi-omics integration remains underexplored.

We present **MoRE**, a framework that aligns heterogeneous assays into a shared latent space by repurposing **frozen pre-trained transformers**.

<p align="center">
  <img src="figures/more_overview.png" alt="MoRE overview" width="800"/>
</p>

---

## Key Idea

Unlike purely generative approaches, **MoRE** prioritizes:

- cross-sample alignment  
- cross-modality alignment  
- structure-preserving representation learning  

instead of relying only on sequence reconstruction.

Specifically, MoRE introduces:

- **modality-specific lightweight adapters**
- **a task-adaptive fusion layer**
- **parameter-efficient fine-tuning (PEFT)** on top of a frozen backbone

<p align="center">
  <img src="figures/more_architecture.png" alt="MoRE architecture" width="800"/>
</p>

---

## Training Objectives

MoRE is optimized with a joint objective that combines:

- **masked modeling loss**
- **supervised contrastive loss**
- **batch-invariant alignment loss**

This design enables the model to learn embeddings that generalize across:

- unseen cell types  
- unseen platforms  
- heterogeneous omics assays 
