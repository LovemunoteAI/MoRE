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


## Usage

Run the workflow in the following order:

### 1. Annotation
Open and run:

```bash
code/1.\ annotate.ipynb
```

This notebook handles dataset annotation and preprocessing.

### 2. MoRE Training / Integration
Then open and run:

```bash
code/2.\ run\ MoRE.ipynb
```

This notebook performs model execution, embedding generation, and downstream integration analysis.

### 3. Model Definition
The core model is implemented in:

```bash
code/MoRE_model.py
```

Make sure this file remains in the correct directory so the notebooks can import it properly.
