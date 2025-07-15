import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Paths or model IDs
base_model_id      = "meta-llama/Llama-2-3b-hf"
finetuned_model_dir = "outputs"  # or your HF repo

# Load validation data
val_df = pd.read_csv("data/processed/symptom-disease-validation-dataset.csv")

# Initialize tokenizers and pipelines
base_tokenizer = AutoTokenizer.from_pretrained(base_model_id)
base_model     = AutoModelForCausalLM.from_pretrained(base_model_id, device_map="auto")
base_pipe      = pipeline("text-generation", model=base_model, tokenizer=base_tokenizer, 
                          max_length=128, pad_token_id=base_tokenizer.eos_token_id)

ft_tokenizer   = AutoTokenizer.from_pretrained(finetuned_model_dir)
ft_model       = AutoModelForCausalLM.from_pretrained(finetuned_model_dir, device_map="auto")
ft_pipe        = pipeline("text-generation", model=ft_model, tokenizer=ft_tokenizer, 
                          max_length=128, pad_token_id=ft_tokenizer.eos_token_id)

def get_diagnosis(pipe, symptom_text):
    prompt = f"Symptoms: {symptom_text}[/INST]Assistant:"
    # return first generated sentence up to end token
    outputs = pipe(prompt, num_return_sequences=1, do_sample=False)
    text = outputs[0]["generated_text"]
    # strip off the prompt
    return text.split("Assistant:")[-1].strip()

# Run inference
results = []
for _, row in val_df.iterrows():
    sym = row["text"]   # or your column name
    true_label = row["label"]
    base_pred = get_diagnosis(base_pipe, sym)
    ft_pred   = get_diagnosis(ft_pipe, sym)
    results.append({
        "symptoms": sym,
        "true": true_label,
        "base_pred": base_pred,
        "ft_pred": ft_pred
    })

comp_df = pd.DataFrame(results)

from sklearn.metrics import accuracy_score, top_k_accuracy_score, confusion_matrix, classification_report

# Exact-match accuracy (base vs fine-tuned)
base_acc = accuracy_score(comp_df["true"], comp_df["base_pred"])
ft_acc   = accuracy_score(comp_df["true"], comp_df["ft_pred"])

print(f"Base Model Accuracy:      {base_acc:.3%}")
print(f"Fine-tuned Model Accuracy:{ft_acc:.3%}")

# If you have a fixed label set, compute Top-3 accuracy
# First encode labels as indices
labels = list(comp_df["true"].unique())
label2idx = {lbl:i for i,lbl in enumerate(labels)}

# Create model scores matrix (for simplicity, 1 if matches, else 0)
y_true = [label2idx[t] for t in comp_df["true"]]
y_base = [label2idx.get(p, -1) for p in comp_df["base_pred"]]
y_ft   = [label2idx.get(p, -1) for p in comp_df["ft_pred"]]

print("Base Top-1 Accuracy:", accuracy_score(y_true, y_base))
print("Fine-tuned Top-1:", accuracy_score(y_true, y_ft))

# Confusion matrices
print("Base Model Confusion Matrix")
print(confusion_matrix(y_true, y_base))

print("Fine-tuned Model Confusion Matrix")
print(confusion_matrix(y_true, y_ft))

# Detailed classification report
print("\nFine-tuned Classification Report:")
print(classification_report(y_true, y_ft, target_names=labels))

# Show a few examples where the fine-tuned model corrected the base model
diffs = comp_df[comp_df["base_pred"] != comp_df["ft_pred"]]
sample = diffs.sample(5, random_state=42)

for _, row in sample.iterrows():
    print("Symptoms:  ", row["symptoms"])
    print(" True:     ", row["true"])
    print(" BasePred: ", row["base_pred"])
    print(" FT-Pred:  ", row["ft_pred"])
    print("-" * 40)
