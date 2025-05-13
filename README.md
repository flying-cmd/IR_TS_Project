# Run

## 1. config environment
```
cd CALF
pip install -r requirements.txt

cd moment
pip install -r requirements.txt
```

## 2. Run

### CALF: 
For Physionet 2012, firstly, you need to put P12data folder (provided by raindrop https://figshare.com/articles/dataset/P12_dataset_for_Raindrop/19514341/1?file=34683085) into CALF/datasets folder.

train_P12.slurm is hpc slurm script, you can copy
```
seq_len=2881
model=CALF
pred_len=0

python run.py \
    --root_path ./datasets/P12/ \
    --is_training 1 \
    --task_name classification \
    --model_id P12_$model'_'$seq_len'_'$pred_len \
    --data P12 \
    --data_type irregular \
    --data_split_path ./datasets/P12data/splits/phy12_split3.npy \
    --seq_len $seq_len \
    --label_len 0 \
    --pred_len $pred_len \
    --batch_size 128 \
    --learning_rate 0.00005 \
    --lradj type1 \
    --train_epochs 100 \
    --d_model 768 \
    --n_heads 4 \
    --d_ff 768 \
    --dropout 0.3 \
    --enc_in 41 \
    --c_out 41 \
    --gpt_layer 6 \
    --itr 1 \
    --model $model \
    --r 32 \
    --lora_alpha 64 \
    --lora_dropout 0.1 \
    --patience 10 \
    --task_loss ce \
    --word_embedding_path wte_pca_600.pt \
    --task_w 1.0 \
    --feature_w 0.01 \
    --output_w 1.0 \
    --quantization 0.016 \
    --classif
```
and add some hpc config to run.

If you want to run on other split, please change data_split_path, like change to --data_split_path ./datasets/P12data/splits/phy12_split1.npy for split 1.

For MIMIC, put mimic_classification folder (https://drive.google.com/drive/folders/1-Ygmbr6TQVN4BxCXkllNNZY-xxVX8iAP) into CALF/datasets/MIMIC folder, like CALF/datasets/MIMIC/mimic_classification/... train_MIMIC.slurm is for MIMIC-III. If you want to run on other random seed, please change --seed.

### Moment:
For Physionet, For Physionet 2012, firstly, you need to put P12data folder (provided by raindrop https://figshare.com/articles/dataset/P12_dataset_for_Raindrop/19514341/1?file=34683085) into moment/data folder. Then, you can run P12_1.sh (shell script) / P12_1.slurm (slurm script), P12_2.sh/.slurm, P12_3.sh/.slurm. 

For MIMIC, put mimic_classification folder (https://drive.google.com/drive/folders/1-Ygmbr6TQVN4BxCXkllNNZY-xxVX8iAP) into moment/data/MIMIC folder, like moment/data/MIMIC/mimic_classification/... Running scripts are similar as P12.

### Time-LLM:
I use the models/TimeLLM_4 (modify for classification)
For forcasting, I follow your work and add `config.py`, I just change a little (pass args to Config to change batch size and random seed conveniently).

All dataset prompt located in the `dataset/prompt_bank`, the prompt file name is same with dataset name. To use prompt, declare `--prompt_domain 1` in shell script.

All shell scripts are located in `scripts`. Classification is run_ir_classification.py. Forecasting is run_ir_forecasting.py.

Consider time efficiency and computation resources, as well as fair comparison. I use GPT-2 with 6 layers. Like 
```
llama_layers=6
--llm_model GPT2 \
```

For using more than one gpu: you can reference
```
model_name=TimeLLM_4
train_epochs=100
learning_rate=0.001
llama_layers=6

master_port=2052
num_process=2
batch_size=8
d_model=32
d_ff=128

comment='TimeLLM-PhysioNet'

accelerate launch --multi_gpu --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_ir_classification.py \
  --seed 42 \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/MIMIC/ \
  --model_id MIMIC_1 \
  --model $model_name \
  --data MIMIC \
  --prompt_domain 1 \
  --features M \
  --seq_len 2880 \
  --num_classes 2 \
  --label_len 0 \
  --pred_len 0 \
  --factor 3 \
  --enc_in 96 \
  --dec_in 96 \
  --c_out 96 \
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
  --patch_len 48 \
  --stride 24 \
  --classif \
```

For using one gpu, you can reference
```
model_name=TimeLLM_4
train_epochs=100
learning_rate=0.00000001
llama_layers=6

master_port=2047
num_process=1
batch_size=16
d_model=32
d_ff=128

comment='TimeLLM-PhysioNet'

accelerate launch --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_ir_classification.py \
  --task_name classification \
  --is_training 0 \
  --root_path ./dataset/P12/ \
  --model_id P12_512_96 \
  --model $model_name \
  --data P12 \
  --prompt_domain 1 \
  --features M \
  --seq_len 2881 \
  --num_classes 2 \
  --label_len 0 \
  --pred_len 0 \
  --factor 3 \
  --enc_in 41 \
  --dec_in 41 \
  --c_out 41 \
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
  --patch_len 48 \
  --stride 24 \
  --n 8000 \
  --quantization 0.016 \
  --classif \
  --split_type random
```

Note: the number of gpu you used should be equal to the number of process.

### S2IP-LLM:
All shell scripts are located in the `Irregular/scripts`.
Classification is Irregular/exp/exp_ir_classification. Forecasting is Irregular/exp/exp_ir_forecasting.

For forcasting, I follow your work and add `config.py`, I just change a little (pass args to Config to change batch size and random seed conveniently).

### UniTS:
It need a .ts format file. Reference: https://github.com/mims-harvard/UniTS/blob/main/Tutorial.md.
For classification, I use convert_HAR/MIMIC/P12_to_ts to convert to .ts file.
For forecasting, I just wrote get_forecast_ts_file, but haven't tried it yet.