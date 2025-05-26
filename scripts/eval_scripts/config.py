import torch
import sys
import os
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
torch.cuda.set_device(0)

data_root_path = './sample_data'


"""
WARNING: if use eval_stage2.py, set FLAG_SENSECOACH = True. If use eval_textonly.py, set FLAG_SENSECOACH = False. If use eval_vllm.py, set FLAG_SENSECOACH = False.
"""
FLAG_SENSECOACH = True

# region ablation study default
# USE_SENSOR_PROCESS = FLAG_BASELINE
USE_FULL_SENSOR_SEQUENCE = True
USE_LSTM_LAYER = False
USE_LEARNABLE_MAPPING = True
# endregion

# models
trained_model_path = f"../../outputs/final_model_6_usinglstm_0_usingfulls_1.pth"
learning_rate = 0.00001
num_epochs = 2
top_k = 6
# endregion
