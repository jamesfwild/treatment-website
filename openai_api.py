import openai
import pandas as pd
from collections import defaultdict
from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np
openai.api_key = "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"

def get_openai_response(user_entry):
    prompt = f"You are an oncologist. Your goal is to provide the most appropriate treatment plan while avoiding unnecessary treatments for a patient with: {user_entry}. You will only provide the procedures in your response as a comma separated list. Only recommend treatments that are strongly supported by clinical evidence."
    response = openai.chat.completions.create(
        model = "ft:gpt-4o-2024-08-06:personal:treatment-plan-final:BCrf9ggZ",  
        messages = [
            {"role": "user", "content": prompt}
        ],
        temperature=0.1,  
        max_tokens=500  
    )
    return response.choices[0].message.content

# test_df = pd.read_json('jsonl_files/findings_to_treatment_test.jsonl', lines=True)
# print(test_df.loc[2, 'messages'])
# message_dict = {msg["role"]: msg["content"] for msg in test_df.loc[1, "messages"]}
# user_content = message_dict["user"]
# assistant_content = message_dict["assistant"]

# per_question_f1_scores = []
# all_predictions = []
# all_actuals = []
# per_sample_metrics = []
# total_tp, total_fp, total_fn = 0, 0, 0

# predicted_label_counts = defaultdict(int)  
# actual_label_counts = defaultdict(int)


# for i in range(len(test_df)):
#     message_dict = {msg["role"]: msg["content"] for msg in test_df.loc[i, "messages"]}
#     user_content = message_dict["user"]
#     assistant_content = message_dict["assistant"]
    
#     prediction = get_openai_response(user_content)
#     actual_treatments = set(assistant_content.split(", "))
#     predicted_treatments = set(prediction.split(", "))
#     all_actuals.append(actual_treatments)
#     all_predictions.append(predicted_treatments)
#     for treatment in predicted_treatments:
#         predicted_label_counts[treatment] += 1
#     for treatment in actual_treatments:
#         actual_label_counts[treatment] += 1

#     tp = len(actual_treatments & predicted_treatments)
#     fp = len(predicted_treatments - actual_treatments)
#     fn = len(actual_treatments - predicted_treatments)

#     total_tp += tp
#     total_fp += fp
#     total_fn += fn

#     precision = tp / (tp + fp) if (tp + fp) > 0 else 0
#     recall = tp / (tp + fn) if (tp + fn) > 0 else 0
#     f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

#     per_question_f1_scores.append(f1)
#     per_sample_metrics.append({"precision": precision, "recall": recall, "f1": f1})

#     print(f"Sample {i+1}:")
#     print(f"  User Input: {user_content}")
#     print(f"  Predicted Treatments: {predicted_treatments}")
#     print(f"  Actual Treatments: {actual_treatments}")
#     print(f"  F1 Score: {f1:.4f}")
#     print("-" * 50)

# unique_labels = sorted(set().union(*all_actuals, *all_predictions))
# y_true = np.array([[label in actual for label in unique_labels] for actual in all_actuals], dtype=int)
# y_pred = np.array([[label in predicted for label in unique_labels] for predicted in all_predictions], dtype=int)
# macro_f1 = f1_score(y_true, y_pred, average="macro")
# micro_f1 = f1_score(y_true, y_pred, average="micro")
# micro_precision = precision_score(y_true, y_pred, average="micro")
# micro_recall = recall_score(y_true, y_pred, average="micro")

# print("=" * 50)
# print(f"Macro F1 Score: {macro_f1:.4f}")
# print(f"Micro F1 Score: {micro_f1:.4f}")
# print(f"Micro Precision: {micro_precision:.4f}")
# print(f"Micro Recall: {micro_recall:.4f}")
# print("=" * 50)
# print("\nLabel Frequency Analysis:")
# print("=" * 50)
# print(f"{'Treatment':<30}{'Predicted Count':<20}{'Actual Count':<20}")

# for treatment in sorted(set(predicted_label_counts.keys()) | set(actual_label_counts.keys())):
#     predicted_count = predicted_label_counts[treatment]
#     actual_count = actual_label_counts[treatment]
#     print(f"{treatment:<30}{predicted_count:<20}{actual_count:<20}")
    
# import json
# with open("model_metrics.json", "r") as f:
#         metrics = json.load(f)
        
# metrics['OpenAI'] = {
#         "macro_f1": macro_f1,
#         "micro_f1": micro_f1,
#         "micro_precision": micro_precision,
#         "micro_recall": micro_recall,
#         "mean_f1_per_question": np.mean(per_question_f1_scores),
#         "f1_per_question": per_question_f1_scores
#     }
# with open("model_metrics.json", "w") as f:
#     json.dump(metrics, f)