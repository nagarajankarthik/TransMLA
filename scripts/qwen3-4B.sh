export HF_HOME=/scratch_aisg/SPEC-SF-AISG/cache/huggingface
export TRANFORMERS_CACHE=/scratch_aisg/SPEC-SF-AISG/cache
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
    --collapse 2 \
    --qk-mqa-dim 64 \
    --q-lora-rank 512 \
    --kv-lora-rank 512 \
    --use-qkv-norm \
    --use-original-norm-weights
