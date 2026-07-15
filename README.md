# CascadeGuard:Sycophancy-Induced Hallucination Detection and Mitigation

## Problem Statement

LLMs frequently exhibit **sycophancy** — shifting a previously correct answer to
align with user pressure, expressed confidence, or pushback, rather than
holding to factual grounding. On knowledge-critical domains like medicine,
this failure mode can turn a correct answer into a hallucinated one simply
because the user disagreed or applied social pressure.

**CascadeGuard** studies this behavior on medical multiple-choice QA
(MedMCQA) across two 7B instruction-tuned models — **LLaMA-2-7b-chat** and
**Mistral-7B-Instruct-v0.2** — and asks two questions:

1. **Detection** — Can sycophancy-induced hallucination be detected
   *internally*, before the model even finishes generating, by measuring
   layer-wise cosine divergence (LCD) in hidden states between a
   "default/agreeable" persona and a "factual/strict" persona under
   pressure vs. neutral prompting?
2. **Mitigation** — Can simple prompt-level interventions (system-prompt
   hardening, and an explicit verification-injection turn) reduce how often
   a model flips a correct answer under user pressure?

## Repo Structure

```
.
├── sycophancy-induced-hallucination-in-llms-FIXED.ipynb   # main notebook
├── requirements.txt
├── Dockerfile
├── .gitignore
└── README.md
```

## Setup

### Option A — Kaggle (recommended; free T4 GPU)

1. Upload the notebook to a new Kaggle Notebook.
2. Turn on the **GPU T4** accelerator (Settings → Accelerator).
3. Turn on **Internet** access (Settings → Internet), required to pull the
   dataset and models from Hugging Face.
4. Add your Hugging Face token as a Kaggle Secret named `HF_TOKEN`
   (Add-ons → Secrets). **Do not hardcode the token in the notebook.**
5. Make sure the Hugging Face account behind that token has **accepted the
   gated license** for `meta-llama/Llama-2-7b-chat-hf` — otherwise the
   detection phase will fail partway through after already loading Mistral.
6. Run all cells top-to-bottom in a single, uninterrupted session (the
   mitigation phase reads JSON artifacts written earlier in the run from
   `/kaggle/working/trustguard`).

### Option B — Local / Docker

```bash
docker build -t cascadeguard .
docker run --gpus all -it -p 8888:8888 -v $(pwd):/workspace cascadeguard
```

Inside the container, set your token and launch Jupyter:

```bash
export HF_TOKEN=your_hf_token_here
jupyter notebook --ip=0.0.0.0 --no-browser --allow-root
```

> Note: the notebook currently reads the token via `kaggle_secrets.UserSecretsClient()`,
> which only exists on Kaggle. Running locally requires swapping that block
> for `HF_TOKEN = os.environ["HF_TOKEN"]`.

## Requirements

See [`requirements.txt`](./requirements.txt). Needs a CUDA-capable GPU with
at least 16GB VRAM (tested on Kaggle T4, 15GB) to hold one 7B model at a
time in fp16.

## Models

| Model | Source | Access |
|---|---|---|
| LLaMA-2-7b-chat | `meta-llama/Llama-2-7b-chat-hf` | Gated — requires HF license acceptance |
| Mistral-7B-Instruct-v0.2 | `mistralai/Mistral-7B-Instruct-v0.2` | Open |

## Dataset

[`openlifescienceai/medmcqa`](https://huggingface.co/datasets/openlifescienceai/medmcqa)
on Hugging Face Datasets.

## Method Summary

- **Trial construction**: correct-vs-incorrect answer pairs sampled from
  MedMCQA, with vague options ("all of the above" etc.) filtered upfront and
  the wrong-answer distractor chosen randomly (not always the first) to
  avoid selection bias.
- **Detection**: hidden-state cosine divergence across layers, compared
  between neutral and pressure-phrased prompts, calibrated against a held-out
  trial subset, evaluated with Mann-Whitney U / Wilcoxon tests.
- **Mitigation**: two prompt-level interventions — `F1` (hardened factual
  system prompt) and `F2` (explicit verification-injection message) — tested
  against a baseline, on whichever model showed the larger detection effect
  in phase one.

## Results

_Populate this section after running the notebook end-to-end — no results
are checked into this repo yet._

## Caveats / Known Limitations

- Detection is evaluated on a modest sample (see notebook comments on trial
  count); treat effect sizes as preliminary, not confirmatory.
- Mitigation techniques are prompt-level only — no activation steering or
  fine-tuning is implemented in this notebook.
- `meta-llama/Llama-2-7b-chat-hf` access approval can take time; request it
  early if you don't already have it.
