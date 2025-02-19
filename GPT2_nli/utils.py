
import numpy as np
from datasets import load_dataset, concatenate_datasets
import evaluate

def load_and_preprocess_data(train_dataset_name, test_dataset_name, tokenizer, data_seed=0, max_length=128):
    def preprocess_function(examples):
        inputs = [f"premise: {p} hypothesis: {h}" for p, h in zip(examples['premise'], examples['hypothesis'])]
        model_inputs = tokenizer(inputs, max_length=max_length, truncation=True, padding="max_length")
        model_inputs["labels"] = examples["label"]
        return model_inputs

    if train_dataset_name == "snli":
        dataset = load_dataset("snli", 'plain_text')
    
    if train_dataset_name == "mnli":
        dataset = load_dataset("glue", "mnli")
    
    encoded_test_datasets = {}

    if "snli" in test_dataset_name:
        snli_dataset = load_dataset("snli", 'plain_text')
        test_dataset = snli_dataset["test"].shuffle(seed=data_seed).select(range(1000))
        test_dataset = test_dataset.filter(lambda x: x["label"] >= 0 and x["label"] <= 2)
        encoded_test_datasets["snli"] = test_dataset.map(preprocess_function, batched=True)
    
    if "mnli_match" in test_dataset_name:
        mnli_dataset = load_dataset("glue", "mnli")
        test_matched = mnli_dataset["validation_matched"].shuffle(seed=data_seed).select(range(1000))
        test_matched = test_matched.filter(lambda x: x["label"] >= 0 and x["label"] <= 2)
        encoded_test_datasets["mnli_match"] = test_matched.map(preprocess_function, batched=True)
    
    if "mnli_mismatch" in test_dataset_name:
        mnli_dataset = load_dataset("glue", "mnli")
        test_mismatched = mnli_dataset["validation_mismatched"].shuffle(seed=data_seed).select(range(1000))
        test_mismatched = test_mismatched.filter(lambda x: x["label"] >= 0 and x["label"] <= 2)
        encoded_test_datasets["mnli_mismatch"] = test_mismatched.map(preprocess_function, batched=True)

    if "anli" in test_dataset_name: 
        anli_dataset = load_dataset("anli")
        anli_test_datasets = anli_dataset["test_r1"].shuffle(seed=data_seed).select(range(1000))
        anli_test_datasets = anli_test_datasets.filter(lambda x: x["label"] >= 0 and x["label"] <= 2)
        encoded_test_datasets["anli"] = anli_test_datasets.map(preprocess_function, batched=True)
    
    train_dataset = dataset["train"].shuffle(seed=data_seed).select(range(8000))
    
    dev_dataset = dataset["validation"].shuffle(seed=data_seed).select(range(1000))
    
    train_dataset = train_dataset.filter(lambda x: x["label"] >= 0 and x["label"] <= 2)
    dev_dataset = dev_dataset.filter(lambda x: x["label"] >= 0 and x["label"] <= 2)
    
    encoded_train_dataset = train_dataset.map(preprocess_function, batched=True)
    encoded_dev_dataset = dev_dataset.map(preprocess_function, batched=True)

    return encoded_train_dataset, encoded_dev_dataset, encoded_test_datasets


def define_compute_metrics():
    accuracy_metric = evaluate.load("accuracy", trust_remote_code=True)
    f1_metric = evaluate.load("f1", trust_remote_code=True)
    recall_metric = evaluate.load("recall", trust_remote_code=True)

    def compute_metrics(p):
        preds = (p.predictions[0]).argmax(-1)
        acc = accuracy_metric.compute(predictions=preds, references=p.label_ids)
        f1 = f1_metric.compute(predictions=preds, references=p.label_ids, average="weighted")
        recall = recall_metric.compute(predictions=preds, references=p.label_ids, average="weighted")
        return {"accuracy": acc["accuracy"], "f1": f1["f1"], "recall": recall["recall"]}
    
    return compute_metrics


if __name__ == '__main__':
    dataset = load_dataset("xnli")
    train_dataset = dataset["train"].select(range(100))
    dev_dataset = dataset["valid"].select(range(100))
    test_dataset = dataset["test"].select(range(100))