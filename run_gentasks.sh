CUDA_VISIBLE_DEVICES=0 python finetune.py \
  --base_model 'yahma/llama-7b-hf' \
  --data_path 'ft-training_set/math_10k.json' \
  --output_dir './trained_models/llama-lora' \
  --batch_size 4 \
  --micro_batch_size 2 \
  --num_epochs 3 \
  --load_8bit \
  --learning_rate 3e-4 \
  --cutoff_len 256 \
  --val_set_size 120 \
  --adapter_name sparseft \
  --sparseft_type 2 \
  --target_modules ["q_proj","k_proj","v_proj","o_proj"] \
  --budget 1000000 --recovery_steps 0

#"gate_proj","up_proj","down_proj",
CUDA_VISIBLE_DEVICES=0 python evaluate.py --model LLaMA-7B --adapter LoRA --dataset gsm8k --base_model 'yahma/llama-7b-hf' --lora_weights './trained_models/llama-lora'