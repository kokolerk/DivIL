import torch
import argparse
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification, TrainingArguments, DataCollatorWithPadding
# import wandb
import json
import csv

from trainer import NLITrainer, TrainEvaluationCallback
from utils import load_and_preprocess_data, define_compute_metrics

torch.manual_seed(0)

def train_and_evaluate(args):
    print("train dataset: ", args.train_dataset_name)
    print("test dataset: ", args.test_dataset_list)
    # wandb.init(project="nli-irm", name=args.run_name)
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token  

    train_dataset, dev_dataset, test_datasets = load_and_preprocess_data(args.train_dataset_name, args.test_dataset_list, tokenizer, args.data_seed, max_length=args.max_length)
    train_dataset_sub = train_dataset.select(range(1000))

    model = GPT2ForSequenceClassification.from_pretrained(args.model_name, num_labels=3, output_hidden_states=True)
    model.config.pad_token_id = tokenizer.eos_token_id  

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="steps",
        eval_steps=500,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.train_epochs,
        weight_decay=0.01,
        report_to="none",
        # report_to="wandb"
    )

    compute_metrics = define_compute_metrics()

    irm_trainer = NLITrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        irm_loss_scale=args.irm_loss_scale,  
        irm_penalty_weight=args.irm_penalty_weight,  
        cl_penalty_weight=args.cl_penalty_weight,  
        ssl_penalty_weight=args.ssl_penalty_weight
    )
    callback = TrainEvaluationCallback(trainer=irm_trainer)
    irm_trainer.add_callback(callback)

    irm_trainer.train()


    csv_header = ["Evaluation Set", "IRM Penalty", "CL Penalty", "SSL Penalty", "Accuracy", "F1", "Recall"]

    with open("evaluation_results.csv", "a", newline='') as csvfile:
        writer = csv.writer(csvfile)
        if csvfile.tell() == 0:
            writer.writerow(csv_header)
        train_results = irm_trainer.evaluate(train_dataset_sub, metric_key_prefix="train")
        writer.writerow([
            "Training Set", 
            args.irm_penalty_weight, 
            args.cl_penalty_weight, 
            args.ssl_penalty_weight, 
            train_results["train_accuracy"],
            train_results["train_f1"],
            train_results["train_recall"]
        ])

        dev_results = irm_trainer.evaluate()
        writer.writerow([
            "Validation Set", 
            args.irm_penalty_weight, 
            args.cl_penalty_weight, 
            args.ssl_penalty_weight, 
            dev_results["eval_accuracy"],
            dev_results["eval_f1"],
            dev_results["eval_recall"]
        ])
        for test_dataset_name, test_dataset in test_datasets.items():
            test_results = irm_trainer.predict(test_dataset)
            test_metrics = compute_metrics(test_results)
            writer.writerow([
                f"Test Set {test_dataset_name}", 
                args.irm_penalty_weight, 
                args.cl_penalty_weight, 
                args.ssl_penalty_weight, 
                test_metrics["accuracy"],
                test_metrics["f1"],
                test_metrics["recall"]
            ])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate a GPT-2 model with IRM")
    parser.add_argument("--model_name", type=str, default="openai-community/gpt2", help="Name of the pre-trained GPT-2 model")
    parser.add_argument("--train_dataset_name", type=str, default="snli", help="Name of the train dataset")
    parser.add_argument("--test_dataset_list", type=str, nargs="+", default=["mnli", "anli"], help="List of test dataset names")
    parser.add_argument("--data_seed", type=int, default=0, help="Random seed for data splitting")
    parser.add_argument("--max_length", type=int, default=128, help="Maximum sequence length for the tokenizer")
    parser.add_argument("--irm_loss_scale", type=float, default=1.0, help="Scale factor for the IRM loss")
    parser.add_argument("--irm_penalty_weight", type=float, default=1.0, help="Penalty weight for the IRM loss")
    parser.add_argument("--cl_penalty_weight", type=float, default=1.0, help="Penalty weight for the contrastive loss")
    parser.add_argument("--ssl_penalty_weight", type=float, default=1.0, help="Penalty weight for the self-supervised contrastive loss")
    parser.add_argument("--train_epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate for the optimizer")
    parser.add_argument("--train_batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--eval_batch_size", type=int, default=8, help="Batch size for evaluation")

    parser.add_argument("--output_dir", type=str, default="./output", help="Output directory for the model")
    parser.add_argument("--run_name", type=str, default="gpt2-irm-experiment", help="Name of the WandB run")

    args = parser.parse_args()
    train_and_evaluate(args)
