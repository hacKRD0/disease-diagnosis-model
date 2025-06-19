import hydra
from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training

@hydra.main(config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    print("Configuration:\n", OmegaConf.to_yaml(cfg))

    # Load tokenizer & base model
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.hf_checkpoint)  :contentReference[oaicite:3]{index=3}
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.hf_checkpoint,
        torch_dtype=cfg.model.model_kwargs.torch_dtype,
        device_map=cfg.model.model_kwargs.device_map,
    )

    # Prepare for LoRA / int8 training
    model = prepare_model_for_int8_training(model)  
    lora_cfg = LoraConfig(
        r=cfg.peft.r,
        lora_alpha=cfg.peft.alpha,
        target_modules=cfg.peft.target_modules,
        lora_dropout=cfg.peft.dropout,
        modules_to_save=cfg.peft.modules_to_save,
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_cfg)  :contentReference[oaicite:4]{index=4}

    # ... load data, define Trainer, etc.
    training_args = TrainingArguments(**cfg.training.args)
    trainer = Trainer(model=model, tokenizer=tokenizer, **vars(training_args))
    trainer.train()

if __name__ == "__main__":
    main()
