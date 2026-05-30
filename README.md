# mmWave-QA

### Can Language Models Understand mmWave Data? Benchmarking Large Language Models for mmWave Radar-Based Human Understanding

> 📌 **Accepted to CVPR 2026 (Findings).**

**Jeongwan Shin**, **Jaehyeon Kim**, **Donguk Ko**, **Jaeho Choi**
DGIST · KAIST InnoCORE LLM
🌐 Project page: https://mmwave-qa.github.io/

---

## Overview

**mmWave-QA** is the **first benchmark for language-conditioned mmWave human perception**. Conventional mmWave systems require per-dataset retraining whenever the hardware, environment, or subject distribution changes. We instead ask: *can off-the-shelf LLMs reason about human motion directly from mmWave radar data, without any radar-specific training?*

Our key idea is a **minimal textualization interface** that serializes each mmWave point cloud into concise natural language — every point becomes `[x, y, z, doppler, intensity]` — so that any off-the-shelf LLM can answer human-motion questions in a **zero-shot, training-free** question-answering (QA) setting.

## Highlights

- **First mmWave–language benchmark.** Aggregates three heterogeneous public mmWave datasets (**mmBody**, **MM-Fi**, **mRI**) and harmonizes them via calibration-aware preprocessing and a **global taxonomy** (139 raw action labels → **86 unified categories**).
- **Training-free LLM perception.** Textualized radar point clouds let frozen LLMs interpret human motion zero-shot — no fine-tuning, no modality-specific encoder.
- **Broad coverage.** **6 real-world scenarios** (normal, furnished, rain, smoke, dark, occlusion) across multiple radar hardware devices (TI IWR1443 / IWR6843, Phoenix-type board).
- **5 complementary QA tasks** probing spatial, temporal, and limb-level reasoning:
  | Task | Description |
  | --- | --- |
  | **ActRec** | Action Recognition |
  | **TrajCheck** | Spatial Movement Verification (centroid moved?) |
  | **ActOrder** | Temporal Action Sequencing |
  | **ActNum** | Action Cardinality Estimation (how many actions) |
  | **LimbFocus** | Primary Limb Motion Detection |
- **Extensive evaluation.** GPT-4o, GPT-5, and Gemini 2.5 (flash/pro) under zero-shot, few-shot, CoT, and few-shot-CoT prompting.

## Key Findings

- **LLMs can reason over radar zero-shot.** Performance improves consistently from zero-shot → few-shot → CoT → few-shot-CoT, with **few-shot-CoT best overall** (GPT-5 leading).
- **mmWave is more robust than RGB under visual degradation.** Under **dark** and **occlusion** conditions, radar-based QA surpasses RGB (GPT-4o), while RGB hallucinates over corrupted frames; the two are comparable in rain/smoke.
- **Temporal balance matters.** Accuracy peaks around **16 frames** — too few limits temporal context, too many adds redundant/noisy points.
- **Hardware quality matters.** The high-resolution Phoenix board yields the best accuracy; noisier low-res sensors (IWR6843/IWR1443) degrade performance.

---

## Repository Layout

- `mmWave-QA_generation.py` – core generator that turns `mm_actions.json` style annotations into QA pairs for the five tasks.
- `script/run_mmwave_qa.sh` – ready-to-run examples covering common generation scenarios.
- `processor.py` – benchmark runner that feeds QA prompts to an LLM and records responses.
- `mmWaveQA_benchmark/` – canonical metadata such as `mm_actions.json` plus train/test splits and per-type QA (`type/`, `etc/`).
- `data/` – placeholder for raw mmWave radar (`radar/`) and aligned video (`video/`) assets.

## Generating QA Benchmarks

The generator consumes an actions JSON file and writes QA samples per task.

```bash
python mmWave-QA_generation.py \
  --actions-json /path/to/mm_actions.json \
  --tasks actionqa,movement,count,temporal,bodymove \
  --output-format single \
  --output-root ./outputs \
  --options 4 \
  --seed 17
```

Key arguments:

| Flag | Description |
| --- | --- |
| `--actions-json` | Path to the mmWave action annotations (defaults to `mmWaveQA_benchmark/mm_actions.json`). |
| `--tasks` | Comma-separated subset of `actionqa,movement,count,temporal,bodymove` or `all`. |
| `--output-root` | Directory that will receive per-task folders or single JSON files. |
| `--output-format` | `files` saves a JSON per segment; `single` aggregates each task into one JSON. |
| `--options` | Number of multiple-choice answers for ActionQA (min 2). |
| `--count-comb-len`, `--temporal-comb-len` | Nominal window sizes (internal logic currently generates fixed ranges). |
| `--seed` | RNG seed for reproducible shuffling. |

For quick starts, adapt the presets in `script/run_mmwave_qa.sh`.

### Output

- When `output-format=files`, each task gets `output-root/<task>/<segment>/...json`.
- When `output-format=single`, each task yields `output-root/<task>.json` with:
  ```json
  {
    "task": "movement",
    "question_input": ["template..."],
    "QA-pairs": [
      {
        "segment": "sequence_01",
        "frames": {"start": 0, "end": 120},
        "question_type": "binary",
        "answer": "yes",
        "...": "..."
      }
    ]
  }
  ```
- Logs summarize how many QA pairs were emitted per task.

## Running the Benchmark

`processor.py` replays benchmark prompts against an LLM.

1. Open the file and set the constants near the top:
   - API key and `MODEL`
   - `INPUT_FILE` (e.g., the aggregated QA JSON)
   - `OUTPUT_FILE`
   - Dataset base paths (`MMBODY_BASE_PATH`, `MMFI_BASE_PATH`, `MRI_BASE_PATH`) if you plan to load point clouds.
2. Run:
   ```bash
   python processor.py \
     --input results_pro/filtered_benchmark_responses.json \
     --output results_pro/gpt5_responses.json \
     --uniform-frames 16 \
     --request-interval 1.0
   ```
3. Optional flags:
   - `--max-entries N` – cap the number of QA items processed.
   - `--test-each-dataset` – ensure one sample per dataset (mmBody, mmFi, MRI).
   - `--enable-reasoning` – trigger rationale (CoT) generation mode if supported by the model.

The processor writes a combined JSON response file and logs progress under `log/benchmark_processor.log`.

## Citation

If you find mmWave-QA useful, please cite:

```bibtex
@inproceedings{shin2026mmwaveqa,
  title     = {Can Language Models Understand mmWave Data? Benchmarking Large Language Models for mmWave Radar-Based Human Understanding},
  author    = {Shin, Jeongwan and Kim, Jaehyeon and Ko, Donguk and Choi, Jaeho},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Findings},
  year      = {2026}
}
```
