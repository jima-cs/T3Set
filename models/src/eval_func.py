import csv
import json
import os

import torch
import torch.nn as nn
def topk_coverage_gpu(preds: torch.Tensor, targets: torch.Tensor, k=3):

    _, topk_indices = preds.topk(k, dim=1)  # [N, k]

    target_scores = torch.gather(targets, 1, topk_indices)  # [N, k]

    correct = target_scores.sum()
    total = targets.sum(dim=1).clamp(min=1e-8)

    precision = correct / (k * preds.size(0))
    recall = correct / total.sum()
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    return precision, recall, f1

def topk_ndcg_gpu(preds: torch.Tensor, targets: torch.Tensor, k=3):

    ideal_sorted = targets.sort(descending=True, dim=1)[0][:, :k]


    _, pred_order = preds.sort(descending=True, dim=1)
    gathered = torch.gather(targets, 1, pred_order[:, :k])


    discounts = 1.0 / torch.log2(torch.arange(k, device=preds.device) + 2.0)
    dcg = (gathered * discounts).sum(dim=1)
    idcg = (ideal_sorted * discounts[:ideal_sorted.size(1)]).sum(dim=1)

    ndcg = (dcg / idcg.clamp(min=1e-8)).mean()
    return ndcg


def save_results_to_json(file_path, results):

    if not os.path.isfile(file_path):
        with open(file_path, 'w') as json_file:
            json.dump([], json_file)

    with open(file_path, 'r') as json_file:
        data = json.load(json_file)

    data.append(results)

    with open(file_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)


def save_results_to_csv(file_path, results, fieldnames):
    if not os.path.isfile(file_path):
        with open(file_path, 'w', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()

    with open(file_path, 'a', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writerow(results)