# Latent-Masked-Video-Models-for-Temporal-Progression
This is a github repository for latent masked video prediction using a frozen VideoMAE encoder. A lightweight transformer predicts future latent states from partial context, showing strong temporal structure and improved performance with masking, especially for structured time-lapse progression.

> **TL;DR:** We train a lightweight transformer to predict future video frames in the latent space of a frozen VideoMAE encoder — no pixel reconstruction, just latent prediction. We find that the encoder's representations are strongly geometrically structured for temporal prediction, and that masking of context tokens provides a statistically significant improvement over full-context prediction, with the benefit being largest for structured biological progressions and smallest for stochastic meteorological ones.


## Table of Contents

- [Motivation](#motivation)
- [What This Project Is (and Is Not)](#what-this-project-is-and-is-not)
- [Dataset](#dataset)
- [Method](#method)
  - [Backbone: VideoMAE](#backbone-videomae)
  - [Architecture](#architecture)
  - [Masking Strategy](#masking-strategy)
  - [Loss Function](#loss-function)
- [Results](#results)
  - [Main Results](#main-results)
  - [Linear Probe: Category Classification](#linear-probe-category-classification)
  - [Ablation: Masked vs Unmasked Context](#ablation-masked-vs-unmasked-context)
- [Figures](#figures)
- [Reproducing the Results](#reproducing-the-results)
- [Key Findings](#key-findings)
- [Limitations](#limitations)

