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