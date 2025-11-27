export HF_HOME=/data/projects/71001002/nkarthik/cache_karthik
export TRANFORMERS_CACHE=/data/projects/71001002/nkarthik/cache_karthik
model_path=Qwen/Qwen2.5-7B-Instruct
save_path=outputs/qwen2_5-7B-Instruct-deepseek
eval_batch_size=8

python transmla/converter.py \
    --model-path $model_path \
    --save-path $save_path \
    --freqfold 4 \
    --ppl-eval-batch-size $eval_batch_size
