CUDA_VISIBLE_DEVICES=1 python -u run.py \
             --task_name long_term_forecast \
             --is_training 1 \
             --root_path ./data/electricity \
             --data_path electricity.csv \
             --model_id electricity_512_96 \
             --model S2IPLLM \
             --data ECL \
             --features M \
             --seq_len 512 \
             --label_len 0 \
             --pred_len 96 \
             --des 'Exp' \
             --itr 1 \
             --d_model 768 \
             --learning_rate 0.0001 \
             --patch_size 16 \
             --stride 8 \
             --add_prompt 1 \
             --prompt_length 4 \
             --batch_size 1024 \
             --sim_coef -0.1 \
             --pool_size 1000 \
             --period 24 \
             --percent 100 \
             --trend_length 24 \
             --seasonal_length 4



CUDA_VISIBLE_DEVICES=1 python -u run.py \
             --task_name long_term_forecast \
             --is_training 1 \
             --root_path ./data/electricity \
             --data_path electricity.csv \
             --model_id electricity_512_192 \
             --model S2IPLLM \
             --data ECL \
             --features M \
             --seq_len 512 \
             --label_len 0 \
             --pred_len 192 \
             --des 'Exp' \
             --itr 1 \
             --d_model 768 \
             --learning_rate 0.0001 \
             --patch_size 16 \
             --stride 8 \
             --add_prompt 1 \
             --prompt_length 4 \
             --batch_size 1024 \
             --sim_coef -0.1 \
             --pool_size 1000 \
             --period 24 \
             --percent 100 \
             --trend_length 24 \
             --seasonal_length 4




CUDA_VISIBLE_DEVICES=1 python -u run.py \
             --task_name long_term_forecast \
             --is_training 1 \
             --root_path ./data/electricity \
             --data_path electricity.csv \
             --model_id electricity_512_336 \
             --model S2IPLLM \
             --data ECL \
             --features M \
             --seq_len 512 \
             --label_len 0 \
             --pred_len 336 \
             --des 'Exp' \
             --itr 1 \
             --d_model 768 \
             --learning_rate 0.0001 \
             --patch_size 16 \
             --stride 8 \
             --add_prompt 1 \
             --prompt_length 4 \
             --batch_size 1024 \
             --sim_coef -0.1 \
             --pool_size 1000 \
             --period 24 \
             --percent 100 \
             --trend_length 24 \
             --seasonal_length 4


CUDA_VISIBLE_DEVICES=1 python -u run.py \
             --task_name long_term_forecast \
             --is_training 1 \
             --root_path ./data/electricity \
             --data_path electricity.csv \
             --model_id electricity_512_720 \
             --model S2IPLLM \
             --data ECL \
             --features M \
             --seq_len 512 \
             --label_len 0 \
             --pred_len 720 \
             --des 'Exp' \
             --itr 1 \
             --d_model 768 \
             --learning_rate 0.0001 \
             --patch_size 16 \
             --stride 8 \ 
             --add_prompt 1 \
             --prompt_length 4 \
             --batch_size 1024 \
             --sim_coef -0.1 \
             --pool_size 1000 \
             --period 24 \
             --percent 100 \
             --trend_length 24 \
             --seasonal_length 4

