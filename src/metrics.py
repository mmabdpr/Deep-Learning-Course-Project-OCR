import torch
import difflib
from src.utils import label2char


def calculate_accuracy_str(y_true, y_pred):
    res = []
    for s1, s2 in zip(y_true, y_pred):
        res.append(matching_blocks(s1, s2) / len(s1))
    avg_acc = sum(res) / len(res)
    return avg_acc


def calculate_accuracy(y_true, y_pred):
    res = []
    y_true = y_true.cpu().detach().tolist()
    for s1, s2 in zip(y_true, y_pred):
        s1 = ''.join([label2char[c] for c in s1])
        s2 = ''.join([label2char[c] for c in s2])
        res.append(matching_blocks(s1, s2) / len(s1))
    return sum(res) / len(res)


def matching_blocks(s1, s2):
    matches = difflib.SequenceMatcher(None, s1, s2).get_matching_blocks()
    total_len = 0
    for match in matches:
        total_len += match.size
    return total_len
