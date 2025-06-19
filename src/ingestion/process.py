from datasets import load_dataset

# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("dux-tecblic/symptom-disease-dataset")

ds.save_to_disk("../data/raw")

# Save the raw dataset to a csv file
print(ds)
