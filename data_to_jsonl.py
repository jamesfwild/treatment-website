from datasets import load_dataset
import json
import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.preprocessing import MultiLabelBinarizer
from skmultilearn.model_selection import iterative_train_test_split
import numpy as np

df = pd.read_csv('dataset.csv')
df = df.dropna(subset=['node_status', 'tnm_tumor', 'tumor_grade']).reset_index(drop=True)
df["final_treatment_plan_binarize"] = df["final_treatment_plan"].apply(lambda x: x.split(","))
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(df["final_treatment_plan_binarize"])
X = np.array(df.index).reshape(-1, 1)

X_train, y_train, X_temp, y_temp = iterative_train_test_split(X, y, test_size=0.3)
X_val, y_val, X_test, y_test = iterative_train_test_split(X_temp, y_temp, test_size=0.33)

ds = DatasetDict({
    "train": Dataset.from_pandas(df.iloc[X_train.flatten()]),
    "validation": Dataset.from_pandas(df.iloc[X_val.flatten()]),
    "test": Dataset.from_pandas(df.iloc[X_test.flatten()])
})

def create_text_representation(example):
    return {
        "text": f"Tumor size: {example['tumor_size']}, Grade: {example['tumor_grade']}, "
                f"Node status: {example['node_status']}, Metastasis: {example['metastasis']}, "
                f"TNM: {example['tnm_tumor']}, ER: {example['er_status']}, "
                f"PR: {example['pr_status']}, HER2: {example['her2_status']}, "
                f"Ki-67: {example['ki_67_categorised']}"
    }

ds = ds.map(create_text_representation)

with open("jsonl_files/findings_to_treatment.jsonl", "w", encoding="utf-8") as f:
    for example in ds['train']:
        json_record = {
            "messages": [
                {"role": "user", "content": f"Results: {example['text']}"},
                {"role": "assistant", "content": example["final_treatment_plan"]}
            ]
        }
        f.write(json.dumps(json_record) + "\n")

with open("jsonl_files/findings_to_treatment_validation.jsonl", "w", encoding="utf-8") as f:
    for example in ds['validation']:
        json_record = {
            "messages": [
                {"role": "user", "content": f"Results: {example['text']}"},
                {"role": "assistant", "content": example["final_treatment_plan"]}
            ]
        }
        f.write(json.dumps(json_record) + "\n")

with open("jsonl_files/findings_to_treatment_test.jsonl", "w", encoding="utf-8") as f:
    for example in ds['test']:
        json_record = {
            "messages" : [
                {"role": "user", "content": f"Results: {example['text']}"},
                {"role": "assistant", "content": example["final_treatment_plan"]}
            ]
        }
        f.write(json.dumps(json_record) + "\n")