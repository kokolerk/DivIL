## Requirements overview

Our implementation relies on the [transformers](https://github.com/huggingface/transformers) package and [PyTorch](https://pytorch.org/).

- python == 3.8.19
- torch == 1.12.0+cu113
- transformers == 4.44.2
- datasets == 3.0.0
- evaluate == 0.4.3
- numpy == 1.24.4

## Procedure

Install this repository and the dependencies using pip:
```bash
$ conda create --name nli python=3.8.19
$ conda activate nli
$ cd GPT2_nli
$ pip install -r requirements.txt
```

## NLI with pretrained GPT-2 Model

### Main results
To reproduce the results from our papers

```bash
python main.py \
        --model_name "openai-community/gpt2" \
        --train_dataset_name $train_dataset_name \
        --test_dataset_list $test_dataset_list \
        --max_length 64 \
        --irm_loss_scale 1.0 \
        --irm_penalty_weight $irm_weight \
        --cl_penalty_weight $cl_weight \
        --ssl_penalty_weight $ssl_weight \
        --train_epochs 5 \
        --learning_rate 2e-5 \
        --train_batch_size 32 \
        --eval_batch_size 32 \
        --output_dir $output_dir \
        --run_name "./output"
```

To build an OOD environment, please set `train_dataset_name` to `snli`, and the test set uses `mnli_match` and `mnli_mismatch`, so that model will be tested in both ID and OOD conditions. Results will be printed at the end of the script, also will be reported to `.csv` file.