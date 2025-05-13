import argparse
import numpy as np
import pathlib
import torch
import torch.nn as nn

from config import Config
from data_loader_full import load_forecasting_data
from aeon.datasets import write_to_ts_file

parser = argparse.ArgumentParser(description="Time Series")
parser.add_argument("--seed", type=int, help="random seed")
parser.add_argument("--problemname", required=True, type=str, help="dataset problemname")
parser.add_argument("--out_dir", default="ts_outputs", type=str, help="Folder for generated .ts file")
parser.add_argument("--out_filename", required=True, type=str, help="file name for the generated .ts file")

args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load config
config = Config(args)

data_obj = load_forecasting_data(config)
train_loader = data_obj["train_dataloader"]
val_loader = data_obj["val_dataloader"]
test_loader = data_obj["test_dataloader"]

for batch_idx, batch in enumerate(test_loader):
    # Get data
    seen_data, seen_tp, seen_mask = batch['observed_data'], batch['observed_tp'], batch['observed_mask'],
    future_data, future_tp, future_mask = batch['data_to_predict'], batch['tp_to_predict'], batch['mask_predicted_data']  
    # print('checking data shape', seen_data.shape, future_data.shape)
    ## prepare the data for moment model
    n_channels = seen_data.shape[2] 


    seen_data = seen_data.permute(0, 2, 1).to(device).float()
    future_data = future_data.permute(0, 2, 1).to(device).float()
    future_mask = future_mask.permute(0, 2, 1).to(device).float()

    # pad to 512 
    seen_data = torch.nn.functional.pad(seen_data, (0, 512 - seen_data.shape[2]), mode='constant', value=0)
    future_data = torch.nn.functional.pad(future_data, (0, 512 - future_data.shape[2]), mode='constant', value=0)
    future_mask = torch.nn.functional.pad(future_mask, (0, 512 - future_mask.shape[2]), mode='constant', value=0)

    # Store original dimensions
    batch_size, n_channels, seq_len = seen_data.shape

    # concat train and test
    full_data = torch.cat([seen_data, future_data], dim=2)

    header = {
        "problemname":  args.problemname,
        "timestamps":   False,
        "missing":      False,
        "univariate":   full_data.shape[1] == 1,
        "equallength":  True,
        "serieslength": full_data.shape[2],   # required when equallength = true
        "targetlabel":  False
    }
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / args.out_filename

    write_to_ts_file(full_data, path=out_dir, problem_name=out_file.stem, header=header, regression=True)
