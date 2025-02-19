import torch
from torch import nn
from transformers import Trainer, TrainerCallback

from loss import IRMLoss, ContrastiveLoss, SelfContrastiveLoss

def get_sequence_lengths(input_ids, pad_token_id):
    if input_ids is not None:
        # if no pad token found, use modulo instead of reverse indexing for ONNX compatibility
        sequence_lengths = torch.eq(input_ids, pad_token_id).int().argmax(-1) - 1
        sequence_lengths = sequence_lengths % input_ids.shape[-1]
        sequence_lengths = sequence_lengths.to(input_ids.device)
    else:
        sequence_lengths = -1

    return sequence_lengths

class NLITrainer(Trainer):
    def __init__(self, mask_fill = 0.7, irm_loss_scale=1.0, irm_penalty_weight=1.0, cl_penalty_weight=1.0, ssl_penalty_weight=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.irm_loss_scale = irm_loss_scale

        self.cls_loss_fn = nn.CrossEntropyLoss()
        self.irm_penalty_weight = irm_penalty_weight
        self.irm_loss_fn = IRMLoss(scale=self.irm_loss_scale)
        self.cl_penalty_weight = cl_penalty_weight
        self.contrastive_loss_fn = ContrastiveLoss(temperature=0.07)

        self.ssl_penalty_weight = ssl_penalty_weight
        self.self_contrastive_loss_fn = SelfContrastiveLoss(temperature=0.07)

        self.mask_fill = mask_fill 
        hidden_dim = self.model.config.hidden_size
        self.proj = torch.nn.Sequential(torch.nn.Linear(hidden_dim, 2 * hidden_dim), 
                    # torch.nn.BatchNorm1d(2 * flags.hidden_dim),
                    torch.nn.ReLU(),
                    torch.nn.Linear(2 * hidden_dim, hidden_dim)).to(self.model.device)

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        loss = self.cls_loss_fn(logits, labels)
        if self.irm_penalty_weight != 0:
            loss += self.irm_loss_fn(logits, labels, self.irm_penalty_weight)
        if self.cl_penalty_weight != 0:
            batch_size = inputs['input_ids'].shape[0]
            sequence_lengths = get_sequence_lengths(inputs['input_ids'], model.config.pad_token_id)
            sentence_feature = outputs.hidden_states[-1][torch.arange(batch_size, device=self.model.device), sequence_lengths]
            loss += self.contrastive_loss_fn(sentence_feature, labels, self.cl_penalty_weight)
        if self.ssl_penalty_weight != 0:
            # [a,b,c] -> [a,a,b,b,c,c]
            cse_inputs = {k: torch.repeat_interleave(v, 2, dim=0) for k, v in inputs.items()}
            cse_batch_size = cse_inputs['input_ids'].shape[0]
            cse_outputs = model(**cse_inputs)
            cse_sequence_lengths = get_sequence_lengths(cse_inputs['input_ids'], model.config.pad_token_id)

            cse_features = cse_outputs.hidden_states[-1][torch.arange(cse_batch_size, device=self.model.device), cse_sequence_lengths]
            cse_features = self.proj(cse_features)

            mask_fill = int(cse_features.shape[1] * self.mask_fill)
            cse_features[:, :mask_fill] = 0
            loss += self.self_contrastive_loss_fn(cse_features, self.ssl_penalty_weight)
        return (loss, outputs) if return_outputs else loss
    
    def evaluate_train(self, train_dataset=None):
        if train_dataset is None:
            train_dataset = self.train_dataset
        train_metrics = self.evaluate(eval_dataset=train_dataset, metric_key_prefix="train")
        return train_metrics
    
class TrainEvaluationCallback(TrainerCallback):
    def __init__(self, trainer):
        self.trainer = trainer

    def on_step_end(self, args, state, control, **kwargs):
        pass
        # if state.global_step % args.eval_steps == 0 and state.global_step > 0:
        #     print(f"\nStep {state.global_step}: Evaluating training set...")
        #     train_metrics = self.trainer.evaluate_train()
        #     print(f"Step {state.global_step}: Evaluating validation set...")
        #     dev_metrics = self.trainer.evaluate()


