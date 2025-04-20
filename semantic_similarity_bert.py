from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer
from skmultilearn.model_selection import iterative_train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score

tokenizer = AutoTokenizer.from_pretrained("neuml/pubmedbert-base-embeddings")
model = AutoModel.from_pretrained("neuml/pubmedbert-base-embeddings")
def get_semantic_prediction():
    pass 
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
def meanpooling(output, mask):
    embeddings = output.last_hidden_state
    mask = mask.unsqueeze(-1).expand(embeddings.size()).float()
    return torch.sum(embeddings * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)

symptoms = [entry["text"] for entry in ds["train"]]
diagnoses = [entry["final_treatment_plan"] for entry in ds["train"]]

diagnosis_inputs = tokenizer(symptoms, padding=True, truncation=True, return_tensors='pt')

with torch.no_grad():
    diagnosis_outputs = model(**diagnosis_inputs)

diagnosis_embeddings = meanpooling(diagnosis_outputs, diagnosis_inputs['attention_mask'])
train_embeddings_np = torch.nn.functional.normalize(diagnosis_embeddings, p=2, dim=1).numpy()
def get_semantic_prediction(symptoms):
    inputs = tokenizer(symptoms, truncation=True, padding="max_length", max_length=512, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = meanpooling(outputs, inputs['attention_mask'])
    embeddings_np = torch.nn.functional.normalize(embeddings, p=2, dim=1).numpy()
    similarities = cosine_similarity(embeddings_np, train_embeddings_np)
    best_match_idx = np.argmax(similarities)
    predicted_treatment_plan = diagnoses[best_match_idx]
    
    return predicted_treatment_plan

# test_texts = [entry["text"] for entry in ds["test"]]
# true_labels = [entry["final_treatment_plan"] for entry in ds["test"]]

# test_inputs = tokenizer(test_texts, padding=True, truncation=True, return_tensors='pt')

# with torch.no_grad():
#     test_outputs = model(**test_inputs)

# test_embeddings = meanpooling(test_outputs, test_inputs['attention_mask'])
# train_embeddings_np = torch.nn.functional.normalize(diagnosis_embeddings, p=2, dim=1).numpy()
# test_embeddings_np = torch.nn.functional.normalize(test_embeddings, p=2, dim=1).numpy()

# predicted_labels = []
# correct_counts = [] 
# total_counts = []
# precision_scores = []
# recall_scores = []
# per_question_f1_scores  = []
# total_tp, total_fp, total_fn = 0, 0, 0

# for i, test_emb in enumerate(test_embeddings_np):
#     similarities = cosine_similarity([test_emb], train_embeddings_np)[0]
#     best_match_idx = np.argmax(similarities)
#     predicted_labels.append(diagnoses[best_match_idx])
    
#     actual_treatments = set(true_labels[i].split(", "))
#     predicted_treatments = set(predicted_labels[i].split(", "))  

#     tp = len(actual_treatments & predicted_treatments)
#     fp = len(predicted_treatments - actual_treatments)
#     fn = len(actual_treatments - predicted_treatments)

#     total_tp += tp
#     total_fp += fp
#     total_fn += fn

#     precision = tp / (tp + fp) if (tp + fp) > 0 else 0
#     recall = tp / (tp + fn) if (tp + fn) > 0 else 0
#     f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

#     per_question_f1_scores.append(f1)
#     print(f"Test Sample {i+1}:")
#     print(f"  Actual Diagnosis: {true_labels[i]}")
#     print(f"  Predicted Diagnosis: {predicted_labels[i]}")
#     print(f"  Precision: {precision:.2f}, Recall: {recall:.2f}, F1-Score: {f1:.2f}")
#     print(similarities[best_match_idx])
#     print("-" * 80)

# y_true = np.array([[label in actual for label in mlb.classes_] for actual in true_labels], dtype=int)
# y_pred = np.array([[label in predicted for label in mlb.classes_] for predicted in predicted_labels], dtype=int)

# micro_precision = precision_score(y_true, y_pred, average="micro")
# micro_recall = recall_score(y_true, y_pred, average="micro")
# micro_f1 = f1_score(y_true, y_pred, average="micro")
# macro_f1 = f1_score(y_true, y_pred, average="macro")

# from scipy.stats import pearsonr

# correlations = [pearsonr(y_true[:, i], y_pred[:, i])[0] for i in range(y_true.shape[1])]
# mean_pearson = np.nanmean(correlations) 

# print(f"Mean Pearson Correlation across labels: {mean_pearson:.4f}")

# print("=" * 80)
# print(f"Micro-Averaged Precision: {micro_precision:.4f}")
# print(f"Micro-Averaged Recall: {micro_recall:.4f}")
# print(f"Micro-Averaged F1 Score: {micro_f1:.4f}")
# print("-" * 80)
# print(f"Macro-Averaged Precision: {macro_f1:.4f}")
# print(f"Pearson: {mean_pearson}")
# print("=" * 80)

# import json
# with open("model_metrics.json", "r") as f:
#         metrics = json.load(f)
        
# metrics['Semantic-Similarity'] = {
#         "macro_f1": macro_f1,
#         "micro_f1": micro_f1,
#         "micro_precision": micro_precision,
#         "micro_recall": micro_recall,
#         "mean_f1_per_question": np.mean(per_question_f1_scores),
#         "f1_per_question": per_question_f1_scores
#     }
# with open("model_metrics.json", "w") as f:
#     json.dump(metrics, f)