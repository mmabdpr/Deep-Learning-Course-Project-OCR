import torch
import difflib
label2char = {0: '', 1: "0", 2: "1", 3: "2", 4: "3", 5: "4",
              6: "5", 7: "6", 8: "7", 9: "8", 10: "9", 11: "-"}


def calculate_accuracy(y_true, y_pred):
    res = []
    y_true = y_true.cpu().detach().tolist()
    for S1, S2 in zip(y_true, y_pred):
        res.append(matching_blocks(S1, S2) / len(S1))
    return sum(res) / len(res)


def matching_blocks(s1, s2):
    s1 = ''.join([label2char[c] for c in s1])
    s2 = ''.join([label2char[c] for c in s2])
    matches = difflib.SequenceMatcher(None, s1, s2).get_matching_blocks()
    total_len = 0
    for match in matches:
        total_len += match.size
    return total_len
