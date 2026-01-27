title: Baseline
summary: Overview & Benchmarking of baseline world models.
sidebar_title: Baseline
---

## DINO World-Model

DINO World-Model (DINO-WM) is a self-supervised latent world model introduced by [Zhou et al., 2025](https://arxiv.org/pdf/2411.04983). To avoid learning from scratch and collapse, DINO-WM leverages frozen DINOv2 features to produce visual observation embedding. The model extract patch-level features from a pretrained DINOv2 encoder and trains a latent dynamics model (predictor) to predict future states in the DINO feature space. Optimal actions are determined at test-time by performing planning with the [Cross-Entropy Method](https://www.iro.umontreal.ca/~lecuyer/myftp/papers/handbook13-ce.pdf) (CEM)

### Training Objective

The model is trained with a teacher-forcing loss, i.e. an l2-loss between the predicted next state embedding $\hat{z}_{t+1}$ and the ground-truth embedding $z_{t+1}$:

$$ \mathcal{L}_{\text{DINOWM}} = \mathcal{L}_{\text{sim}} = \| \hat{z}_{t+1} - z_{t+1} \|_2^2 $$

where $z_{t+1}$ represents the frozen DINOv2 features of the next observation.

### Benchmark

!!! danger ""
    Evaluation is performed with fixed 50 steps budget, unlike the infinite budget of the original paper

| Environment | Success Rate | Checkpoint |
|-------------|--------------|------------|
| [Push-T](envs/pusht.md) | ~86% | NA |



## Planning with Latent Dynamics Model

Planning with Latent Dynamics Model (PLDM) is a Joint-Embedding Predictive Architecture (JEPA) proposed by [Sobal et al., 2025](https://arxiv.org/pdf/2502.14819). Unlike DINO-WM which relies on frozen pretrained features, PLDM trains the encoder and predictor jointly from scratch using a combination of losses to prevent representational collapse:

- $\mathcal{L}_{\text{sim}}$: teacher-forcing loss between predicted and target embeddings
- $\mathcal{L}_{\text{std}}$, $\mathcal{L}_{\text{cov}}$: [variance-covariance regularization](https://arxiv.org/pdf/2105.04906) (VCReg) to maintain embedding diversity
- $\mathcal{L}_{\text{temp}}$: temporal smoothness regularizer between consecutive embeddings
- $\mathcal{L}_{\text{idm}}$: inverse dynamics modeling loss to predict actions from embedding pairs

Optimal actions are search at test-time with planning by leveraging the [Model Path Predictive Integral](https://acdslab.github.io/mppi-generic-website/docs/mppi.html) (MPPI) solver.

### Training Objective

$$ \mathcal{L}_{\text{PLDM}} = \mathcal{L}_{\text{sim}} + \alpha \mathcal{L}_{\text{std}} + \beta \mathcal{L}_{\text{cov}} + \delta \mathcal{L}_{\text{temp}} + \omega \mathcal{L}_{\text{idm}}$$

where $\alpha$, $\beta$, $\delta$, and $\omega$ are hyperparameters controlling the contribution of each regularization term.

### Benchmark

| Environment | Success Rate | Checkpoint |
|-------------|--------------|------------|
| ? | ? | NA |


## Goal-Conditioned Behavioural Cloning 

| Environment | Success Rate | Checkpoint |
|-------------|--------------|------------|
| ? | ? | NA |


## Inverse Q-Learning

| Environment | Success Rate | Checkpoint |
|-------------|--------------|------------|
| ? | ? | NA |
