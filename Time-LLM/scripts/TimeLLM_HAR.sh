model_name=TimeLLM_4
train_epochs=100
learning_rate=0.0001
llama_layers=6

master_port=2047
num_process=1
batch_size=64
d_model=32
d_ff=128

comment='TimeLLM-Activity'

accelerate launch --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_ir_classification.py \
  --seed 123 \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/activity/ \
  --model_id Activity_1 \
  --model $model_name \
  --data activity \
  --prompt_domain 1 \
  --features M \
  --seq_len 50 \
  --num_classes 7 \
  --label_len 0 \
  --pred_len 0 \
  --factor 3 \
  --enc_in 12 \
  --dec_in 12 \
  --c_out 12 \
  --des 'Exp' \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --llm_model GPT2 \
  --llm_dim 768 \
  --llm_layers $llama_layers \
  --train_epochs $train_epochs \
  --model_comment $comment \
  --patch_len 8 \
  --stride 4 \
  --classif \
  --classify-pertp \

#accelerate launch --multi_gpu --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
#  --task_name long_term_forecast \
#  --is_training 1 \
#  --root_path ./dataset/ETT-small/ \
#  --data_path ETTh1.csv \
#  --model_id ETTh1_512_192 \
#  --model $model_name \
#  --data ETTh1 \
#  --features M \
#  --seq_len 512 \
#  --label_len 48 \
#  --pred_len 192 \
#  --factor 3 \
#  --enc_in 7 \
#  --dec_in 7 \
#  --c_out 7 \
#  --des 'Exp' \
#  --itr 1 \
#  --d_model 32 \
#  --d_ff 128 \
#  --batch_size $batch_size \
#  --learning_rate 0.02 \
#  --llm_layers $llama_layers \
#  --train_epochs $train_epochs \
#  --model_comment $comment
#
#accelerate launch --multi_gpu --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
#  --task_name long_term_forecast \
#  --is_training 1 \
#  --root_path ./dataset/ETT-small/ \
#  --data_path ETTh1.csv \
#  --model_id ETTh1_512_336 \
#  --model $model_name \
#  --data ETTh1 \
#  --features M \
#  --seq_len 512 \
#  --label_len 48 \
#  --pred_len 336 \
#  --factor 3 \
#  --enc_in 7 \
#  --dec_in 7 \
#  --c_out 7 \
#  --des 'Exp' \
#  --itr 1 \
#  --d_model $d_model \
#  --d_ff $d_ff \
#  --batch_size $batch_size \
#  --lradj 'COS'\
#  --learning_rate 0.001 \
#  --llm_layers $llama_layers \
#  --train_epochs $train_epochs \
#  --model_comment $comment
#
#accelerate launch --multi_gpu --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
#  --task_name long_term_forecast \
#  --is_training 1 \
#  --root_path ./dataset/ETT-small/ \
#  --data_path ETTh1.csv \
#  --model_id ETTh1_512_720 \
#  --model $model_name \
#  --data ETTh1 \
#  --features M \
#  --seq_len 512 \
#  --label_len 48 \
#  --pred_len 720 \
#  --factor 3 \
#  --enc_in 7 \
#  --dec_in 7 \
#  --c_out 7 \
#  --des 'Exp' \
#  --itr 1 \
#  --d_model $d_model \
#  --d_ff $d_ff \
#  --batch_size $batch_size \
#  --learning_rate $learning_rate \
#  --llm_layers $llama_layers \
#  --train_epochs $train_epochs \
#  --model_comment $comment
