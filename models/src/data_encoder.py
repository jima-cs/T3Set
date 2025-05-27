import torch
import torch.nn as nn

from key_dim import PREDEFINE_pose_dim_keys
from setting import USE_FULL_SENSOR_SEQUENCE


class SensorEncoder(nn.Module):
    def __init__(self, sensor_input_size, output_dim=64):
        super(SensorEncoder, self).__init__()
        self.output_dim = output_dim
        self.encoder = nn.Sequential(
            nn.Linear((100 if USE_FULL_SENSOR_SEQUENCE else 1), 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Linear(128, output_dim),
            nn.Dropout(0.2)
        )
    def __calculating_static_feature(self, x):
        pass
    def forward(self, x):
        batch_size = x.size(0)
        stroke_num = x.size(1)
        sensor_key_dim_nums = x.size(2)
        sensor_windows_double = x.size(3)
        x = x.view(batch_size * stroke_num * sensor_key_dim_nums, -1)
        encoded = self.encoder(x)
        return encoded.view(batch_size, stroke_num, sensor_key_dim_nums, -1)


class PoseEncoder(nn.Module):
    def __init__(self, pose_input_size, output_dim=64):
        super(PoseEncoder, self).__init__()
        self.output_dim = output_dim
        self.joint_features_size = 3
        self.num_joints = len(PREDEFINE_pose_dim_keys)
        self.num_frames = 10

        self.encoder = nn.Sequential(
            nn.Linear(self.joint_features_size, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Linear(128, output_dim),
            nn.Dropout(0.2)
        )

    def forward(self, x):
        batch_size, stroke_num, num_joints, num_frames, _ = x.shape
        x = x.view(batch_size * stroke_num * num_joints * num_frames, -1)
        encoded = self.encoder(x)
        return encoded.view(batch_size, stroke_num, num_joints, num_frames, -1).mean(dim=3)
