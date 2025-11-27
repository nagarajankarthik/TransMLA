export HF_HOME=/data/projects/71001002/nkarthik/cache_karthik
export TRANFORMERS_CACHE=/data/projects/71001002/nkarthik/cache_karthik
model_path=Qwen/Qwen3-4B
save_path=outputs/qwen3-4B-deepseek-qkv-norm
eval_batch_size=8

python3 transmla/converter.py \
    --model-path $model_path \
    --save-path $save_path \
    --cal-dataset wikitext2 \
    --dtype bf16 \
    --device cuda:0 \
    --ppl-eval-batch-size $eval_batch_size \
    --freqfold 2 \
    --collapse 1 \
    --qk-mqa-dim 64 \
    --q-lora-rank 512 \
    --kv-lora-rank 512 \
    --use-qkv-norm \
    --use-original-norm-weights
