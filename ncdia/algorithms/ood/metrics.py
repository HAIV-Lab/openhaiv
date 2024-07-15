import numpy as np
import sklearn.metrics as skm


def ood_metrics(conf, label, tpr_th: float = 0.95):
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

    precision_in, recall_in, th_in = \
        skm.precision_recall_curve(1 - ood_indicator, conf)
    
    precision_out, recall_out, th_out = \
        skm.precision_recall_curve(ood_indicator, -conf)
    
    auroc = skm.auc(fpr_list, tpr_list)
    aupr_in = skm.auc(recall_in, precision_in)
    aupr_out = skm.auc(recall_out, precision_out)

    return fpr, auroc, aupr_in, aupr_out
