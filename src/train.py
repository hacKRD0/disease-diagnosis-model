import json
import hydra
import os
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    # 1) Load the preprocessed dataset
    data_files = {
        "train": os.path.join(get_original_cwd(), cfg.data.data_files.train),
        "validation": os.path.join(get_original_cwd(), cfg.data.data_files.validation)
    }
    # print("data_files -----> ", data_files)
    ds = load_dataset("csv", data_files=data_files, **cfg.data.csv_kwargs)
    

    # 2) Load tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.hf_checkpoint)
    # Ensure tokenizer has a padding token; causal LMs like Llama often don't.
    if tokenizer.pad_token is None:
        # Reuse EOS token as PAD to avoid adding new embeddings
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.hf_checkpoint,
        torch_dtype=cfg.model.model_kwargs.torch_dtype,
        device_map=cfg.model.model_kwargs.device_map,
    )
    # Sync model padding with tokenizer
    model.config.pad_token_id = tokenizer.pad_token_id
    # Prepare and apply LoRA
    model = prepare_model_for_kbit_training(model)
    lora_cfg = LoraConfig(
        r=cfg.model.peft.r,
        lora_alpha=cfg.model.peft.lora_alpha,
        target_modules=cfg.model.peft.target_modules,
        lora_dropout=cfg.model.peft.lora_dropout,
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_cfg)

    # 3) Tokenize on-the-fly
    def tokenize_fn(batch):
        """Prepare combined prompt+answer sequences and align labels.

        Each example becomes:
            "Symptoms: <symptom text>\n\nDisease: <disease> <eos>"
        Prompt tokens are masked out in ``labels`` with ``-100`` so the loss is
        only computed on the answer portion.
        """
        # Build prompt and answer strings
        prompts = [f"Symptoms: {sym}\n\nDisease:" for sym in batch["text"]]
        answers = [str(label) for label in batch["label"]]

        # Full texts fed to the model
        full_texts = [p + " " + a + tokenizer.eos_token for p, a in zip(prompts, answers)]
        tokenized = tokenizer(full_texts, truncation=True, padding="longest")

        # Construct labels that ignore (mask) the prompt part
        labels = []
        for p, a in zip(prompts, answers):
            prompt_ids = tokenizer(p, add_special_tokens=False).input_ids
            answer_ids = tokenizer(" " + a + tokenizer.eos_token, add_special_tokens=False).input_ids
            labels.append([-100] * len(prompt_ids) + answer_ids)

        # Left-pad labels to the max sequence length in this batch so labels
        # shape matches ``input_ids``.
        max_len = max(len(l) for l in labels)
        for l in labels:
            l += [-100] * (max_len - len(l))

        tokenized["labels"] = labels
        return tokenized
    tokenized = ds.map(tokenize_fn, batched=True, remove_columns=ds["train"].column_names)
    # tokenized.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    # 4) Prepare Trainer
    training_args = TrainingArguments(
        output_dir=cfg.training.output_dir,
        num_train_epochs=cfg.training.num_epochs,
        per_device_train_batch_size=cfg.training.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.training.per_device_eval_batch_size,
        learning_rate=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        logging_steps=cfg.training.logging_steps,
        save_steps=cfg.training.save_steps,
        eval_strategy=cfg.training.eval_strategy,
        eval_steps=cfg.training.eval_steps,
        seed=cfg.training.seed,
        dataloader_pin_memory=cfg.training.dataloader_pin_memory,
        label_names=["label"],
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
    )

    # batch = tokenized["train"][:2]
    # print({k: torch.tensor(v).shape for k, v in batch.items()})
    # Expect: {'input_ids': torch.Size([2, seq]), 'attention_mask': [2, seq], 'labels': [2, seq]}


    # 5) Train & save
    trainer.train()
    trainer.save_model(cfg.training.output_dir)

if __name__ == "__main__":
    main()
