import torch
import torch.nn as nn
from model.fusion_model import FusionModel

def load_model(sensor_input_size, pose_input_size,USE_SENSOR_PROCESS, USE_LSTM_LAYER=False, USE_LEARNABLE_MAPPING=True, path='fusion_model.pth'):
    model = FusionModel(sensor_input_size, pose_input_size, USE_SENSOR_PROCESS=USE_SENSOR_PROCESS, USE_LSTM_LAYER=USE_LSTM_LAYER, USE_LEARNABLE_MAPPING=USE_LEARNABLE_MAPPING).to(device)
    model.load_state_dict(torch.load(path))
    return model