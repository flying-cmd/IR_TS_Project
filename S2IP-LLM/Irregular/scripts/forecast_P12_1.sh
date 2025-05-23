CUDA_VISIBLE_DEVICES=0 python -u run.py \
  --seed 42 \
  --train_epochs 2 \
  --task_name ir_forecast \
  --is_training 1 \
  --root_path ./data/forecast/physionet \
  --data_path physionet \
  --model_id physionet_1 \
  --model S2IPLLM \
  --data physionet \
  --number_variable 41 \
  --features M \
  --seq_len 512 \
  --feature_dim 41 \
  --label_len 0 \
  --pred_len 512 \
  --des 'Exp' \
  --itr 1 \
  --d_model 768 \
  --learning_rate 0.0001 \
  --loss 'CE' \
  --patch_size 16 \
  --stride 8 \
  --patience 10 \
  --add_prompt 1 \
  --prompt_length 4 \
  --batch_size 48 \
  --sim_coef -0.05 \
  --pool_size  1000 \
  --percent 100 \
  --trend_length 96 \
  --seasonal_length 96 \
