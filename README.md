# mmWave-QA

End-to-end tooling for generating multi-task question-answer (QA) benchmarks from mmWave radar motion annotations and for evaluating model responses with OpenAI's API.

## Repository Layout

- `mmWave-QA_generation.py` – core generator that turns `mm_actions.json` style annotations into QA pairs for multiple tasks (ActionQA, Movement, Count, Temporal, BodyMove).
- `script/run_mmwave_qa.sh` – ready-to-run examples that cover common generation scenarios.
- `processor.py` – OpenAI-powered benchmark runner that feeds QA prompts to a GPT-style model and records responses.
- `mmWaveQA_benchmark/` – canonical metadata such as `mm_actions.json` plus train/test splits.
- `data/` – placeholder for raw mmWave radar (`radar/`) and aligned video (`video/`) assets.
- `_notebook/Test_API.ipynb` – lightweight notebook for sanity-checking API connectivity and the generator pipeline.

## Requirements

- Python 3.9+
- `pip install tqdm numpy openai`
- Access to mmWave action annotations (see below) and, for response generation, an OpenAI API key with quota.

## Setup

1. Create and activate a virtual environment (optional but recommended).
2. Install dependencies:
   ```bash
   pip install --upgrade pip
   pip install tqdm numpy openai
   ```
3. Place your annotation JSON (matching the schema in `mmWaveQA_benchmark/mm_actions.json`) somewhere accessible. Update paths passed to the generator accordingly.

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

For quick starts, adapt the presets in `script/run_mmwave_qa.sh`. Each block shows the exact CLI invocation to regenerate ActionQA, Movement, Count, Temporal, BodyMove, or all tasks at once.

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

## Processing Benchmarks with OpenAI (Optional)

`processor.py` can replay benchmark prompts against GPT-style models.

1. Open the file and set the constants near the top:
   - `OPENAI_API_KEY`, `MODEL`
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
   - `--enable-reasoning` – trigger rationale generation mode if supported by the model.

The processor writes a combined JSON response file and logs progress under `log/benchmark_processor.log`.