# import sys
import numpy as np
import sklearn.metrics as skm

# def stable_cumsum(arr, rtol=1e-05, atol=1e-08):
#     """Use high precision for cumsum and check that final value matches sum."""
#     out = np.cumsum(arr, dtype=np.float64)
#     expected = np.sum(arr, dtype=np.float64)
#     if not np.allclose(out[-1], expected, rtol=rtol, atol=atol):
#         raise RuntimeError('cumsum was found to be unstable: '
#                            'its last element does not correspond to sum')
#     return out

# def fpr_and_fdr_at_recall(y_true, y_score, recall_level=0.95):
#     """Compute FPR at a given recall level."""
#     y_true = (y_true == 1)  # Treat ID samples (1) as positive
#     desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
#     y_score = y_score[desc_score_indices]
#     y_true = y_true[desc_score_indices]

#     distinct_value_indices = np.where(np.diff(y_score))[0]
#     threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

#     tps = stable_cumsum(y_true)[threshold_idxs]
#     fps = 1 + threshold_idxs - tps

#     recall = tps / tps[-1]
#     cutoff = np.argmin(np.abs(recall - recall_level))

#     return fps[cutoff] / (np.sum(np.logical_not(y_true)))

# def get_measures(in_score, out_score, recall_level=0.95):
#     """Compute AUROC, AUPR, and FPR."""
#     labels = np.concatenate([np.ones(len(in_score)), np.zeros(len(out_score))])
#     scores = np.concatenate([in_score, out_score])

#     auroc = skm.roc_auc_score(labels, scores)
#     aupr_in = skm.average_precision_score(labels, scores)
#     aupr_out = skm.average_precision_score(1 - labels, -scores)
#     fpr = fpr_and_fdr_at_recall(labels, scores, recall_level)

#     return fpr, auroc, aupr_in, aupr_out


def ood_metrics(conf: np.ndarray, label: np.ndarray, tpr_th: float = 0.95):
    """Compute OOD metrics.

    Args:
        conf (np.ndarray): Confidence scores. Shape (N,).
        label (np.ndarray): Label array. Shape (N,). Containing:
            -1: OOD samples.
            int >= 0: ID samples with class labels
        tpr_th (float): True positive rate threshold to compute
            false positive rate.

    Returns:
        fpr (float): False positive rate.
        auroc (float): Area under the ROC curve.
        aupr_in (float): Area under the precision-recall curve
            for in-distribution samples.
        aupr_out (float): Area under the precision-recall curve
            for out-of-distribution samples.
    """
    # following convention in ML we treat OOD as positive
    ood_indicator = np.zeros_like(label)
    ood_indicator[label == -1] = 1

    # in the postprocessor we assume ID samples will have larger
    # "conf" values than OOD samples
    # therefore here we need to negate the "conf" values
    fpr_list, tpr_list, th = skm.roc_curve(ood_indicator, -conf)
    fpr = fpr_list[np.argmax(tpr_list >= tpr_th)]

    precision_in, recall_in, th_in = skm.precision_recall_curve(1 - ood_indicator, conf)

    precision_out, recall_out, th_out = skm.precision_recall_curve(ood_indicator, -conf)

    auroc = skm.auc(fpr_list, tpr_list)
    aupr_in = skm.auc(recall_in, precision_in)
    aupr_out = skm.auc(recall_out, precision_out)

    return fpr, auroc, aupr_in, aupr_out


def search_threshold(conf: np.ndarray, label: np.ndarray, prec_th: float):
    """Search for the threshold for OOD detection.

    Args:
        conf (np.ndarray): Confidence scores. Shape (N,).
        label (np.ndarray): Label array. Shape (N,). Containing:
            -1: OOD samples.
            int >= 0: ID samples with class labels
        prec_th (float): Precision threshold.

    Returns:
        best_th (float): Threshold for OOD detection.
        prec (float): Precision at the threshold.
        recall (float): Recall at the threshold.
    """
    # following convention in ML we treat OOD as positive
    ood_indicator = np.zeros_like(label)
    ood_indicator[label == -1] = 1

    # in the postprocessor we assume ID samples will have larger
    # "conf" values than OOD samples
    # therefore here we need to negate the "conf" values
    precisions, recalls, thresholds = skm.precision_recall_curve(ood_indicator, -conf)

    idx = np.argmax(precisions >= prec_th)
    best_th = thresholds[idx]

    return -best_th, precisions[idx], recalls[idx]
