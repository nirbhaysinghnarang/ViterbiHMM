import numpy as np


class Validator:
    def __init__(self, token_labels, token_indices):
        self.token_labels = token_labels
        self.token_indices = token_indices

    def format_output_labels(self):

        label_dict = {"LOC": [], "MISC": [], "ORG": [], "PER": []}
        prev_label = 'O'
        start = self.token_indices[0]
        for idx, label in enumerate(self.token_labels):
            curr_label = label.split('-')[-1]
            if label.startswith("B-") or (curr_label != prev_label and curr_label != "O"):
                if prev_label != "O":
                    label_dict[prev_label].append(
                        (start, self.token_indices[idx-1]))
                start = self.token_indices[idx]
            elif label == "O" and prev_label != "O":
                label_dict[prev_label].append(
                    (start, self.token_indices[idx-1]))
                start = None

            prev_label = curr_label
        if start is not None:
            label_dict[prev_label].append((start, self.token_indices[idx-1]))
        return label_dict

    def mean_f1(y_pred_dict, y_true_dict):
        F1_lst = []
        for key in y_true_dict:
            TP, FN, FP = 0, 0, 0
            num_correct, num_true = 0, 0
            preds = y_pred_dict[key]
            trues = y_true_dict[key]
            for true in trues:
                num_true += 1
                if true in preds:
                    num_correct += 1
                else:
                    continue
            num_pred = len(preds)
            if num_true != 0:
                if num_pred != 0 and num_correct != 0:
                    R = num_correct / num_true
                    P = num_correct / num_pred
                    F1 = 2*P*R / (P + R)
                else:
                    F1 = 0      # either no predictions or no correct predictions
            else:
                continue
            F1_lst.append(F1)
        return np.mean(F1_lst)
