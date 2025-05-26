import argparse
import math
import time
from datetime import datetime
from typing import List, Set, Tuple
import numpy as np
import torch.nn as nn
import os
import sys

sys.path.append(os.getcwd())

from eval_func import topk_coverage_gpu, topk_ndcg_gpu, save_results_to_json, save_results_to_csv
from key_dim import sensor_phy_dim_keys, PREDEFINE_pose_dim_keys, \
    RoundDataIncludesPoseSensor, PREDEFINE_sensor_dim_keys, all_data_keys
from data_loader import load_round_data, sample_case_path, get_datakey_idx_from_datakey, \
    preprocess_data_as_data_loader
from new_key_mapping import AllowedSuggestionType, AllowedSuggestionKey, clean_key_mapping, \
    suggKey_to_dataKey_mapping, allowed_sugg_keys

from predict_model import FusionModel
import torch
import torch.optim as optim
from setting import device, num_epochs, \
    learning_rate, top_k, USE_SENSOR_PROCESS, USE_FULL_SENSOR_SEQUENCE


def compute_loss(preds: torch.Tensor, targets: torch.Tensor, pos_weight2: torch.Tensor):
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight2)
    return criterion(preds, targets)




def evaluate_v(model, val_loader, device , topk, pos_weight1):
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in val_loader:
            sensor_x, pose_x, stroke_mask, _, _, targets = batch
            sensor_x = sensor_x.to(device, non_blocking=True)
            pose_x = pose_x.to(device, non_blocking=True)
            stroke_mask = stroke_mask.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits = model(sensor_x, pose_x, stroke_mask)
            loss = compute_loss(logits, targets, pos_weight2=pos_weight1)

            probs = torch.sigmoid(logits)
            val_loss += loss.item()
            all_preds.append(probs)
            all_labels.append(targets)
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_labels, dim=0)

    precision, recall, f1 = topk_coverage_gpu(all_preds, all_targets, k=topk)
    ndcg = topk_ndcg_gpu(all_preds, all_targets, k=topk)

    return val_loss / len(val_loader), precision.cpu().item(), recall.cpu().item(), f1.cpu().item(), ndcg.cpu().item()
def train(train_loader, valid_loader, sensor_input_size, pose_input_size, num_classes, num_epochs,
          learning_rate, pos_weight_:float, top_k):
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    fusion_model = FusionModel(sensor_input_size, pose_input_size, USE_SENSOR_PROCESS=USE_SENSOR_PROCESS, USE_LSTM_LAYER=USE_LSTM_LAYER, USE_LEARNABLE_MAPPING=USE_LEARNABLE_MAPPING).to(device)
    fusion_model.apply(init_weights)
    optimizer = torch.optim.AdamW(fusion_model.parameters(), lr=learning_rate, weight_decay=1e-5)
    best_f1 = 0.4
    saved_val_precision, saved_val_recall, saved_val_f1, saved_val_ngcd = 0, 0, 0, 0
    saved_model = None
    for epoch in range(num_epochs):
        fusion_model.train()
        epoch_train_loss = 0.0
        for batch in train_loader:
            sensor_data, pose_data, stroke_mask, _, _, targets = batch
            optimizer.zero_grad()
            logits = fusion_model(sensor_data.to(device), pose_data.to(device), stroke_mask.to(device))

            batch_loss = compute_loss(logits, targets.to(device, non_blocking=True), pos_weight2=torch.tensor([pos_weight_]).to(device))
            batch_loss.backward()
            optimizer.step()
            epoch_train_loss += batch_loss.item()
        train_loss = epoch_train_loss / len(train_loader)
        val_loss, val_precision, val_recall, val_f1, val_ngcd = evaluate_v(fusion_model, valid_loader, device, topk=top_k, pos_weight1=torch.tensor([pos_weight_]).to(device))

        print(f"Epoch {epoch + 1}/{num_epochs} | "
              f"Train Loss: {train_loss / len(train_loader):.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Precision@{top_k}: {val_precision:.4f} | "
              f"Recall@{top_k}: {val_recall:.4f} | "
              f"F1@{top_k}: {val_f1:.4f} | "
              f"NDCG@{top_k}: {val_ngcd:.4f}")
        if val_f1 > best_f1:

            best_f1 = val_f1
            saved_val_precision, saved_val_recall, saved_val_f1, saved_val_ngcd = val_precision, val_recall, val_f1, val_ngcd

    path = f'outputs/final_model_{top_k}_usinglstm_{1 if USE_LSTM_LAYER else 0}_usingfulls_{1 if USE_FULL_SENSOR_SEQUENCE else 0}_usesp_{USE_SENSOR_PROCESS}_uselmap_{USE_LEARNABLE_MAPPING}_lr_5_epochnum_{num_epochs}.pth'
    save_model(fusion_model, sensor_input_size=sensor_input_size, pose_input_size=pose_input_size,
               path=path)
    return fusion_model, (saved_val_precision, saved_val_recall, saved_val_f1, saved_val_ngcd)



def load_model_from_checkpoint(checkpoint_path, sensor_input_size, pose_input_size):
    model = FusionModel(sensor_input_size, pose_input_size, USE_SENSOR_PROCESS=USE_SENSOR_PROCESS, USE_LSTM_LAYER=USE_LSTM_LAYER, USE_LEARNABLE_MAPPING=USE_LEARNABLE_MAPPING)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    checkpoint = torch.load(checkpoint_path)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"Model loaded from {checkpoint_path}, epoch: {epoch}, loss: {loss}")
    return model, optimizer


def save_model(model, sensor_input_size, pose_input_size, path):
    torch.save(model.state_dict(), path)
def load_model(sensor_input_size, pose_input_size,USE_SENSOR_PROCESS=USE_SENSOR_PROCESS, USE_LSTM_LAYER=False, USE_LEARNABLE_MAPPING=True, path='fusion_model.pth'):
    model = FusionModel(sensor_input_size, pose_input_size, USE_SENSOR_PROCESS=USE_SENSOR_PROCESS, USE_LSTM_LAYER=USE_LSTM_LAYER, USE_LEARNABLE_MAPPING=USE_LEARNABLE_MAPPING).to(device)
    model.load_state_dict(torch.load(path))
    return model

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Ablation Study Parameters')

    parser.add_argument('--use_lstm_layer', type=str2bool, default=True,
                        help='Whether to use LSTM layer')

    parser.add_argument('--use_learnable_mapping', type=str2bool, default=True,
                        help='Whether to use learnable mapping')
    parser.add_argument('--learning_rate', type=float, default=0.00001,
                        help='Learning rate for the model')
    parser.add_argument('--num_epochs', type=int, default=180, help='Number of epochs for training')
    parser.add_argument('--top_k', type=int, choices=[5, 6, 7, 8, 9], default=6, help='Top K value for metrics')

    args = parser.parse_args()
    USE_LSTM_LAYER = args.use_lstm_layer
    print('using lstm layer', USE_LSTM_LAYER)
    USE_LEARNABLE_MAPPING = args.use_learnable_mapping
    learning_rate = args.learning_rate
    num_epochs = args.num_epochs
    top_k = args.top_k

    batch_size = 5
    embedding_dim = 64
    fps = 10

    sensor_input_size = len(PREDEFINE_sensor_dim_keys) * (100 if USE_FULL_SENSOR_SEQUENCE else 1)
    pose_input_size = fps * len(PREDEFINE_pose_dim_keys) * 3  # 3是xyz
    num_keys = len(PREDEFINE_sensor_dim_keys) + len(PREDEFINE_pose_dim_keys)


    round_data_list: List[RoundDataIncludesPoseSensor] = load_round_data(sample_case_path)
    train_set, test_set, valid_set = preprocess_data_as_data_loader(round_data_list, batch_size, fps)
    all_labels = []
    for _, _, _, _, _, labels in train_set:
        all_labels.extend(labels.tolist())
    for _, _, _, _, _, labels in test_set:
        all_labels.extend(labels.tolist())
    for _, _, _, _, _, labels in valid_set:
        all_labels.extend(labels.tolist())
    print(len(all_labels))
    num_all_rounds = len(all_labels)
    total_pos = sum(sum(keys) for keys in all_labels)
    total_neg = len(allowed_sugg_keys) * len(all_labels) - total_pos

    pos_weight_f = total_neg / (total_pos + 1e-7)
    print("pos_weight", round(pos_weight_f, 3), round(total_pos / len(all_labels), 3))

    saved_model, saved_val_res = train(train_set, valid_set, sensor_input_size, pose_input_size, num_keys,
                  num_epochs, learning_rate, pos_weight_f, top_k=top_k)
    saved_val_precision, saved_val_recall, saved_val_f1, saved_val_ndcg = saved_val_res
    test_loss, test_precision, test_recall, test_f1, test_ndcg = evaluate_v(saved_model, test_set, device, topk=top_k,
                                                               pos_weight1=torch.tensor([pos_weight_f]).to(device))

    print(f"Test Loss: {test_loss:.4f} | "
          f"Precision@{top_k}: {test_precision:.4f} | "
          f"Recall@{top_k}: {test_recall:.4f} | "
          f"F1@{top_k}: {test_f1:.4f} | "
          f"NDGC@{top_k}: {test_ndcg:.4f}")
    results = {
        'num_round': num_all_rounds,
        'pos_weight': round(pos_weight_f, 3),
        'avg_suggkeys_each_round': round(total_pos / len(all_labels), 3),
        'use_learnable_network': USE_LEARNABLE_MAPPING,
        'use_aggregated_sensor_dim_keys': USE_SENSOR_PROCESS,
        'use_LSTM_layer': USE_LSTM_LAYER,
        'use_full_sensor_sequence': USE_FULL_SENSOR_SEQUENCE,
        'top_k': top_k,
        'learning_rate': learning_rate,
        'epoch_num': num_epochs,
        'test_precision': round(test_precision, 4),
        'test_recall': round(test_recall, 4),
        'test_f1': round(test_f1, 4),
        'test_NDCG': round(test_ndcg, 4),
        'valid_precision': round(saved_val_precision, 4),
        'valid_recall': round(saved_val_recall, 4),
        'valid_f1': round(saved_val_f1, 4),
        'valid_NDCG': round(saved_val_ndcg, 4),
        'time': str(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    }
    json_file_path = 'outputs/ablation_study_results_32_complete.json'
    csv_file_path = 'outputs/ablation_study_results_32_complete.csv'
    save_results_to_json(json_file_path, results)
    save_results_to_csv(csv_file_path, results, list(results.keys()))