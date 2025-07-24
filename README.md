# Disease Diagnosis Fine-Tuning with DistilBERT

This repository demonstrates **end-to-end fine-tuning of a small language model (e.g. `distilbert-base-uncased`) to predict medical diagnoses from free-text symptom descriptions**.  
The workflow is intentionally lightweight – relying only on the HuggingFace ecosystem, [Hydra](https://github.com/facebookresearch/hydra) for configuration, and [LoRA](https://arxiv.org/abs/2106.09685) via `peft` for parameter-efficient training.

---
## 1. Problem Statement
* **Input**: A short natural-language description of a patient’s symptoms.
* **Output**: The most probable disease / condition from a predefined label set.

Rather than training a bespoke classifier, we cast the task as *conditional language modelling*:

```
Symptoms: <symptom text>

Disease: <LABEL>
```

During fine-tuning the model learns to generate the correct label after the `Disease:` prompt.

---
## 2. Repository Structure
| Path | Purpose |
|------|---------|
| `data/raw/` | Original CSVs with numeric `label` ids. |
| `data/processed/` | Cleaned CSVs with textual labels (produced by `preprocess_dataset.py`). |
| `configs/` | Hydra ΩConf files holding all hyper-parameters. |
| `src/ingestion/preprocess_dataset.py` | Converts numeric labels → text labels and saves processed CSVs. |
| `src/train.py` | Finetunes the model using 🤗 **Trainer** + **LoRA**. |
| `src/inference/generate_predictions.py` | Runs inference on the validation set and compares base vs finetuned accuracy. |

---
## 3. Quickstart
### 3.1 Environment
```bash
# Python ≥3.9 recommended
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt  # or install the packages below
```
Minimal dependencies:
```
transformers datasets peft accelerate hydra-core pandas scikit-learn
```

### 3.2 Data Preparation
Place your raw CSVs in `data/raw/` with **two columns**:
* `text`  – symptom description (string)
* `label` – integer class id

Create `data/label_mapping.json` mapping **disease → id** (the script will invert it).

```bash
python -m src.ingestion.preprocess_dataset \ 
  --train_csv data/raw/symptom-disease-train-dataset.csv \ 
  --val_csv   data/raw/symptom-disease-test-dataset.csv  \ 
  --mapping_file data/label_mapping.json
```

`data/processed/` will now contain cleaned train/validation CSVs with textual labels.

### 3.3 Training
Hyper-parameters live in `configs/config.yaml` (learning-rate, batch-size, LoRA rank, etc.).
Simply run:
```bash
python -m src.train
```
Key details inside `src/train.py`:
1. **Dataset loading** – HuggingFace `load_dataset("csv", data_files=...)`.
2. **Prompt building** – `"Symptoms: {sym}\n\nDisease:"` (prompt) + `<label> <eos>` (answer).
3. **Label masking** – tokens corresponding to the prompt are set to `-100` so loss is only computed on the answer.
4. **Model** – loaded with `AutoModelForCausalLM.from_pretrained(cfg.model.hf_checkpoint)` (default can be DistilBERT or any causal model).
5. **LoRA** – applied via `peft.get_peft_model` for memory-efficient fine-tuning.
6. **Training** – 🤗 `Trainer` handles optimisation, evaluation, checkpointing.

Outputs are stored in `outputs/` (configurable).

### 3.4 Inference & Evaluation
After training, compare the base model vs the fine-tuned one:
```bash
python -m src.inference.generate_predictions
```
This script:
* Loads the **validation CSV** from `data/processed/`.
* Builds the same prompt format and generates a label with both the *base* and *fine-tuned* pipelines.
* Computes accuracy, confusion matrix, and a full classification report.

---
## 4. Configuration Highlights (`configs/config.yaml`)
| Field | Example | Meaning |
|-------|---------|---------|
| `model.hf_checkpoint` | `distilbert-base-uncased` | Starting HF model. |
| `model.peft.r` | `8` | LoRA rank. |
| `training.num_epochs` | `3` | Training epochs. |
| `data.data_files.train` | `data/processed/symptom-disease-train-dataset.csv` | Training CSV. |

Edit these values to experiment with different models or batch sizes.

---
## 5. Tips & Notes
* **GPU**: Training DistilBERT with LoRA fits easily on a single 8 GB GPU.
* **New labels**: Add the new disease name ↔ id to `label_mapping.json` and regenerate processed CSVs.
* **Long inputs**: Increase `max_seq_length` in the tokenizer call if symptom texts are lengthy.

---
## 6. Acknowledgements
* 🤗 **Transformers**, **Datasets**, **PEFT**
* **Hydra** for elegant configuration management

Happy experimenting! 🎉