# Sycophancy Causal Effect: A Cross-Family Analysis of Instruction Tuning

> Final project for the **Natural Language Processing** course (Master's Degree in Computer Science / Data Science for Economics and Health), Università degli Studi di Milano, A.Y. 2025-26.
> Topic **P9**: *Measuring total causal effects of instruction tuning on LLM behaviour.*

---

## Overview

This project frames **instruction tuning as a causal "treatment"** and measures its **total effect on sycophancy** — the tendency of language models to agree with a user's stated claim even when that claim is false.

Using a paired experimental design across three model families (Qwen, Llama, Gemma), we ask:

> Does instruction tuning make language models more sycophantic? And is the effect uniform across model families, or does the alignment recipe of each vendor produce qualitatively different behaviours?

We answer with quantitative evidence: **across 7,200 model evaluations on TruthfulQA, the causal effect of instruction tuning on sycophancy is family-dependent in both magnitude and direction**.

---

## Key Findings

All ATEs below are computed on the unified error-rate outcome `p_error` (defined in *Methodology* below): a **positive ATE means instruction tuning makes the model err more often** — more sycophantic at L1–L3, or less accurate at L0. A negative ATE means it improves the model.

- **Statistically significant effects in 11 out of 12 (family × premise-strength) cells**, surviving Bonferroni correction in 8 of them.
- **Qwen2.5-1.5B-Instruct** shows the "classic" sycophancy pattern: substantially less error-prone than its base under weak/medium false premises (ATE ≈ −0.22) but cedes substantially under strong premises (ATE = +0.24). Effect sizes are **extreme** (|Cohen's d_z| up to 4.7).
- **Llama-3.2-1B-Instruct** shows the **opposite** pattern at L3: stronger resistance than the base under strong premises (ATE = −0.20, d_z = −2.10). However, it also exhibits a notable error-rate increase at L0 (ATE = +0.136, d_z = +0.52) — an accuracy regression on the baseline forced-choice format.
- **Gemma-2-2B-it** shows modest mean effects but extreme **per-question polarisation**: distribution of differences is bimodal at L3, peaking near both ±0.4.
- Per-category breakdown reveals systematic patterns: *Stereotypes* function as a universal "safety wall" (all families more resistant than baseline), while *Confusion* and *Logical Falsehood* are universal weak spots, with Gemma reaching the highest sycophantic responses on these categories (ATE up to +0.51).

---

## Methodology in one paragraph

For each of 300 TruthfulQA questions, we generate four prompts that vary the strength of the user's stated false premise (L0 neutral / L1 weak / L2 medium / L3 strong, all with binary A/B answer choices). For every (prompt, model) combination we compute the next-token softmax over the `A` and `B` tokens — a scoring procedure that works identically on base and instruction-tuned models *without* chat templates, isolating the causal effect of instruction tuning from prompt-format confounders. Because the semantic meaning of "choosing A" differs by template (correct answer at L0; *"Yes, the false claim is true"* at L1–L3), we report results on a **unified error-rate outcome** `p_error`, defined as `1 − P(A)` at L0 and `P(A)` at L1–L3 — so higher values always mean the model is more wrong, regardless of level. We then estimate the **Average Treatment Effect** of instruction tuning on `p_error` at each premise strength within each family, with paired t-tests and 10,000-sample bootstrap 95% CIs.

---

## Repository Structure

```
sycophancy-causal-effect/
├── README.md                    ← you are here
├── requirements.txt             ← Python dependencies
├── .gitignore
│
├── src/                         ← reusable Python modules (imported by notebooks)
│   ├── data/
│   │   └── dataset_builder.py   ← prompt templates, TruthfulQA loading, sampling
│   └── models/
│       └── inference.py         ← ModelScorer class, score_agreement function (computes p_agree)
│
├── notebooks/
│   ├── 01_pilot.ipynb           ← inference notebook (run on Colab w/ T4 GPU)
│   └── 02_analysis.ipynb        ← analysis notebook (derives p_error; runs locally, no GPU)
│
├── results/                     ← experiment outputs (committed for reproducibility)
│   ├── main_multifamily_n300.csv             ← main experiment (300 × 4 × 6 = 7200 rows; raw p_agree)
│   ├── main_multifamily_n300.json            ← same data, JSON format
│   ├── agg_mean_perror.csv                   ← aggregated descriptive stats on p_error
│   ├── stats_per_family_level.csv            ← paired t-tests + bootstrap CIs on p_error
│   └── stats_per_category_family_L3.csv      ← per-category breakdown at L3 (p_error ≡ p_agree here)
│
├── figures/                     ← all plots used in the paper, as PDFs
│   ├── fig1_forest_ATE.pdf
│   ├── fig2_interaction_perror.pdf
│   ├── fig3_heatmap_ATE.pdf
│   ├── fig4_scatter_perexample_L3.pdf
│   ├── fig5_histogram_diff_perlevel.pdf
│   └── fig6_heatmap_ATE_by_category.pdf
│
└── paper/                       ← LaTeX source of the report (added in final phase)
```

> **Note on the outcome variable.** The raw scorer in `src/models/inference.py` always reports `p_agree = P(token A)`. The analysis notebook derives `p_error` from `p_agree` as a uniformly-interpretable error-rate metric: `p_error = 1 − p_agree` at L0 (where A is the correct answer), `p_error = p_agree` at L1–L3 (where A confirms the user's false claim). This keeps the inference layer agnostic and makes the analysis side semantically uniform.

---

## How to Reproduce

The pipeline is split into two notebooks: **inference** (needs a GPU, runs on Colab) and **analysis** (runs locally, no GPU).

### 1. Prerequisites

- A Google account (for Colab + Drive)
- A HuggingFace account, with **license accepted** for the gated models:
  - [`meta-llama/Llama-3.2-1B`](https://huggingface.co/meta-llama/Llama-3.2-1B) and `Llama-3.2-1B-Instruct`
  - [`google/gemma-2-2b`](https://huggingface.co/google/gemma-2-2b) and `gemma-2-2b-it`
- A HuggingFace **read token** ([create one here](https://huggingface.co/settings/tokens))
- Python 3.10+ for the local analysis

### 2. Clone the repository

```bash
git clone https://github.com/SamueleSamonini/sycophancy-causal-effect.git
cd sycophancy-causal-effect
```

### 3. Inference (Colab)

1. Open `notebooks/01_pilot.ipynb` in **Google Colab**.
2. Set the runtime to **T4 GPU** (`Runtime → Change runtime type`).
3. Run all cells in order. The first run will:
   - mount Google Drive
   - clone this repo into the Colab session
   - install dependencies from `requirements.txt`
   - prompt you for your HuggingFace token (saved to a private file on Drive for future sessions)
   - download Qwen, Llama, and Gemma model weights to a Drive cache (~3-8 GB total)
   - run inference on 300 TruthfulQA questions × 4 prompt levels × 6 models (~10-15 minutes)
   - save results to Drive as CSV/JSON

4. Download `main_multifamily_n300.csv` and `.json` from Drive and place them in `results/` of your local repo. Push via Git.

### 4. Analysis (local)

```bash
pip install -r requirements.txt
```

Then open `notebooks/02_analysis.ipynb` in VS Code (or Jupyter) with a **local Python kernel** (no GPU needed). Run all cells. The notebook will:

- load `results/main_multifamily_n300.csv`
- derive the unified `p_error` outcome variable from `p_agree`
- compute aggregate statistics, paired t-tests, bootstrap 95% CIs, Cohen's d_z (all on `p_error`)
- generate all figures (saved to `figures/`) and statistical tables (saved to `results/`)
- include per-question and per-category drill-down analyses

---

## Models and Data

| Family | Base | Instruction-tuned |
|---|---|---|
| Qwen | `Qwen/Qwen2.5-1.5B` | `Qwen/Qwen2.5-1.5B-Instruct` |
| Llama | `meta-llama/Llama-3.2-1B` | `meta-llama/Llama-3.2-1B-Instruct` |
| Gemma | `google/gemma-2-2b` | `google/gemma-2-2b-it` |

**Dataset**: [TruthfulQA](https://huggingface.co/datasets/truthful_qa) (`generation` config, 817 validation questions; we sample 300 with fixed seed 42).

---

## Authors

- **Samuele Samonini** — student, Università degli Studi di Milano

---

## AI Usage Disclaimer

Parts of this project have been developed with the assistance of **Anthropic's Claude (Opus 4.7)**. The AI was used as a pair-programming and methodological-design partner across the following dimensions:

- discussion and selection of the research question among the proposed thematic clusters
- design of the experimental pipeline (logit-based scoring, four-level premise-strength moderator, cross-family replication strategy, unified `p_error` outcome variable)
- drafting of code (Python modules in `src/`, notebook scaffolds, plotting code)
- drafting of descriptive text and analysis interpretation

All AI-assisted output has been carefully reviewed, executed, debugged, and validated by the author. The author retains full responsibility for the final content, methodological choices, code correctness, and academic integrity of the project.

---

## License

This project is released for academic and educational purposes.
Model weights, dataset, and underlying libraries are subject to their respective licences (see HuggingFace model cards and dataset pages).