# MODEL_PATH="path/to/your/model" # change to the model you want to evaluate
BASE_MODEL_NAME="meta-llama/Llama-3.2-3B-Instruct" # change to the base model name corresponding to the model you want to evaluate

python eval_countdown.py \
    --model_path "${MODEL_PATH}" \
    --base_model_name "${BASE_MODEL_NAME}" \
    --eval_data_path "data/countdown.json" \
    --eval_samples 2000 \
    --eval_offset -2000 \
    --max_new_tokens 1024 \
    --batch_size 500 \
    --save_responses \
    --show_examples 5