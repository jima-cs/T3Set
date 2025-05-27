import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Set, Dict
from data_encoder import SensorEncoder, PoseEncoder
from new_key_mapping import suggKey_to_dataKey_mapping, allowed_sugg_keys, \
    suggKey_to_dataKey_aggregated_mapping
from key_dim import all_data_keys
from setting import device
class FusionModel(nn.Module):
    def __init__(self, sensor_input_size, pose_input_size, USE_LSTM_LAYER, USE_LEARNABLE_MAPPING, USE_SENSOR_PROCESS, hidden_size=64):
        super().__init__()
        self.sensor_encoder = SensorEncoder(sensor_input_size=sensor_input_size,
                                            output_dim=hidden_size)
        self.pose_encoder = PoseEncoder(pose_input_size,
                                        output_dim=hidden_size)
        self.key_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=4,
            dropout=0.1
        )
        self.USE_LSTM_LAYER = USE_LSTM_LAYER
        self.output = nn.Linear(hidden_size, 1)
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, batch_first=True)
        self._init_weights()
        self.mapping_matrix = nn.Parameter(
            self._build_mapping_matrix(
                (suggKey_to_dataKey_aggregated_mapping if USE_SENSOR_PROCESS else suggKey_to_dataKey_mapping),
                len(all_data_keys))
        ) if USE_LEARNABLE_MAPPING else self._build_mapping_matrix(
            (suggKey_to_dataKey_aggregated_mapping if USE_SENSOR_PROCESS else suggKey_to_dataKey_mapping),
            len(all_data_keys))

    def _init_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.normal_(layer.bias, mean=0, std=0.01)

    def _build_mapping_matrix(self, mapping: Dict[str, List[str]], total_data_keys: int):

        suggkeyid_datakey_id_mapping = {}
        for sugg_key, list_data_key in mapping.items():
            suggkeyid_datakey_id_mapping[allowed_sugg_keys.index(sugg_key)] = [all_data_keys.index(key) for key in list_data_key]
        matrix = torch.zeros(len(suggkeyid_datakey_id_mapping), total_data_keys, device=device)
        for i, (_, data_keys) in enumerate(suggkeyid_datakey_id_mapping.items()):
            matrix[i, data_keys] = 1.0
        return matrix.transpose(0, 1)
    def forward(self, sensor_x, pose_x, stroke_mask):

        sensor_feat = self.sensor_encoder(sensor_x)  # [B, S, sensorK, H]
        pose_feat = self.pose_encoder(pose_x)  # [B, S, poseK, H]

        all_feat = torch.cat([sensor_feat, pose_feat], dim=2)  # [B, S, dataK, H]

        B, S, K, H = all_feat.shape
        if self.USE_LSTM_LAYER:
            lstm_input = all_feat.permute(1, 0, 2, 3).contiguous()
            lstm_input = lstm_input.view(S, B * K, H)

            if stroke_mask is not None:
                lengths = stroke_mask.sum(dim=1)  # [B]
                lengths = lengths.repeat_interleave(K)  # [B*K]
            else:
                lengths = torch.full((B * K,), S, device=lstm_input.device)

            lengths = lengths.clamp(min=1)

            sorted_lengths, sorted_indices = torch.sort(lengths, descending=True)
            sorted_input = lstm_input[:, sorted_indices, :]

            packed_input = nn.utils.rnn.pack_padded_sequence(
                sorted_input, sorted_lengths.cpu(), enforce_sorted=True)
            packed_output, _ = self.lstm(packed_input)
            output, _ = nn.utils.rnn.pad_packed_sequence(packed_input)

            _, original_indices = torch.sort(sorted_indices)
            output = output[:, original_indices, :]  # [S, B*K, H]

            last_indices = (lengths - 1).clamp(min=0)
            batch_key_indices = last_indices[None, :, None].expand(1, -1, H)
            key_feat = output.gather(0, batch_key_indices).squeeze(0)
            key_feat = key_feat.view(B, K, H)  # [B, K, H]
        else:
            if stroke_mask is not None:
                valid = stroke_mask.view(B, S, 1, 1).sum(dim=1)

                key_feat = (all_feat * stroke_mask.view(B, S, 1, 1)).sum(dim=1)
                key_feat /= torch.clamp(valid, min=1e-8)
            else:
                key_feat = all_feat.mean(dim=1)

        attn_in = key_feat.permute(1, 0, 2)  # [K, B, H]
        attn_out, _ = self.key_attention(attn_in, attn_in, attn_in)

        final_feat = attn_out.permute(1, 0, 2)

        data_key_weights = self.output(final_feat).squeeze(-1)
        sugg_key_weights = torch.matmul(data_key_weights, self.mapping_matrix.to(device))
        return sugg_key_weights
