# ShapKO: Shapley-Adaptive Modality Knockout for Robust Multimodal Learning

This repository contains the official implementation of our MICCAI 2026 paper:
**"ShapKO: Shapley-Adaptive Modality Knockout for Robust Multimodal Learning"**

**Accepted at MICCAI 2026**

<p align="center">
  <img src="Figure/Fig1.png" alt="ShapKO overview" width="100%">
</p>

---

## 📄 Paper

> Nusrat Binta Nizam, Fengbei Liu, Sunwoo Kwak, Minh Nguyen, Ruining Deng, Mert R. Sabuncu.
> *ShapKO: Shapley-Adaptive Modality Knockout for Robust Multimodal Learning.*
> Accepted at MICCAI 2026.
> [Full Paper](#)  <!-- add arXiv / proceedings link here -->

Multimodal medical models often degrade when inputs are missing, a common
scenario in real clinical workflows. Even when all modalities are present,
*modality dominance* leads optimization to over-rely on the most predictive
modality and undertrain complementary sources. **ShapKO** periodically estimates
each modality's importance via **Shapley values** over validation subsets and
raises the knockout probability of dominant modalities (a *drop-strong-more*
rule), promoting complementary representations with **no architectural changes**.

---

## ⚙️ Method

ShapKO alternates between two phases:

- **Phase 1 — train under knockout.** Each present modality `m` is kept with
  probability `1 - r_m` (knocked out otherwise, at least one retained).
  Knocked-out and structurally-missing embeddings are replaced by fixed
  placeholders before fusion, and the model is trained on the task loss.
- **Phase 2 — adapt rates (every `K` epochs).** With the model frozen, ShapKO
  evaluates a scalar utility `v(S)` per modality subset on validation, estimates
  Shapley importances, and updates the per-modality knockout rates.
  
---

## 🗂️ Repository Structure

```
shapko/
├── shapko/
│   ├── shapley.py     # exact + Monte-Carlo Shapley (Eq. 1)
│   ├── knockout.py    # base rate, simplex weights, rate update, mask + apply (Eqs. 2-4)
│   ├── utility.py     # subset enumeration + validation-utility estimation
│   ├── metrics.py     # AUC, multilabel AUC, Cox C-index
│   ├── models.py      # MultimodalModel interface + reference model
│   └── trainer.py     # two-phase ShapKOTrainer
├── assets/            # figures used in this README
├── configs/           # per-task hyperparameters
├── examples/          # verify_core.py, toy_example.py
├── tests/             # unit tests for the core math
├── pyproject.toml
├── requirements.txt
└── LICENSE
```


## 📝 Citation

If you find this work useful, please consider citing us:

```bibtex
@inproceedings{nizam2026shapko,
  title     = {ShapKO: Shapley-Adaptive Modality Knockout for Robust Multimodal Learning},
  author    = {Nizam, Nusrat Binta and Liu, Fengbei and Kwak, Sunwoo and Nguyen, Minh and Deng, Ruining and Sabuncu, Mert R.},
  booktitle = {Medical Image Computing and Computer-Assisted Intervention (MICCAI)},
  year      = {2026}
}
```

---

## 📬 Contact

For questions or issues, reach out to: 📧 nn284@cornell.edu

## 🙏 Acknowledgments

This work is funded by NewYork-Presbyterian, NYP–Cornell Cardiovascular AI
Collaboration.
