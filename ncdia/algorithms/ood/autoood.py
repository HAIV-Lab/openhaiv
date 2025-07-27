from ncdia.utils import ALGORITHMS
from .inference import (
    mls_inf,
    msp_inf,
    mcm_inf,
    glmcm_inf,
    dpm_inf,
    neglabel_inf,
    energy_inf,
    vim_inf,
    dml_inf,
    dmlp_inf,
    prot_inf,
    klm_inf,
    # she_inf,
    # relation_inf
)


class AutoOOD(object):
    """AutoOOD class for evaluating OOD detection methods."""

    def __init__(self) -> None:
        super(AutoOOD, self).__init__()

    @staticmethod
    def eval(
        prototype_cls,
        fc_weight,
        train_feats,
        train_logits,
        id_feats,
        id_logits,
        id_labels,
        ood_feats,
        ood_logits,
        ood_labels,
        metrics: list = [],
        tpr_th: float = 0.95,
        prec_th: float = None,
        id_attrs=None,
        ood_attrs=None,
        prototype_att=None,
    ) -> dict:
        """Evaluate the OOD detection methods and return OOD scores.

        Args:
            prototype_cls (np.ndarray): prototype of training data
            fc_weight (np.ndarray): weight of the last layer
            train_feats (np.ndarray): feature of training data
            train_logits (np.ndarray): logits of training data
            id_feats (np.ndarray): feature of ID data
            id_logits (np.ndarray): logits of ID data
            id_labels (np.ndarray): labels of ID data
            ood_feats (np.ndarray): feature of OOD data
            ood_logits (np.ndarray): logits of OOD data
            ood_labels (np.ndarray): labels of OOD data
            metrics (list, optional): list of OOD detection methods to evaluate. Defaults to [].
            tpr_th (float, optional): True positive rate threshold. Defaults to 0.95.
            prec_th (float, optional): Precision threshold. Defaults to None.

        Returns:
            dict: OOD scores, keys are the names of the OOD detection methods,
                values are the OOD scores and search threshold.
                Each value is a tuple containing the following
                - ood metrics (tuple):
                    - fpr (float): false positive rate
                    - auroc (float): area under the ROC curve
                    - aupr_in (float): area under the precision-recall curve for in-distribution samples
                    - aupr_out (float): area under the precision-recall curve for out-of-distribution samples
                - search threshold (tuple): threshold for OOD detection if prec_th is not None
                    - best_th (float): best threshold for OOD detection
                    - conf (torch.Tensor): confidence scores
                    - label (torch.Tensor): label array
                    - precisions (float): precision when precisions >= prec_th
                    - recalls (float): recall when precisions >= prec_th

        """
        ood_scores = {}
        for metric in metrics:
            ood_scores[metric] = ALGORITHMS[metric](
                id_gt=id_labels,
                id_logits=id_logits,
                id_feat=id_feats,
                ood_gt=ood_labels,
                ood_logits=ood_logits,
                ood_feat=ood_feats,
                train_logits=train_logits,
                train_feat=train_feats,
                fc_weight=fc_weight,
                prototypes=prototype_cls,
                tpr_th=tpr_th,
                prec_th=prec_th,
            )
            # if metric == 'msp':
            #     ood_scores['msp'] = msp(id_labels, id_logits, ood_labels, ood_logits, tpr_th, prec_th)
            # elif metric == 'mcm':
            #     ood_scores['mcm'] = mcm(id_labels, id_logits, ood_labels, ood_logits, tpr_th, prec_th)
            # elif metric == 'maxlogit':
            #     ood_scores['maxlogit'] = max_logit(id_labels, id_logits, ood_labels, ood_logits, tpr_th, prec_th)
            # elif metric == 'energy':
            #     ood_scores['energy'] = energy(id_labels, id_logits, ood_labels, ood_logits, tpr_th, prec_th)
            # elif metric == 'vim':
            #     ood_scores['vim'] = vim(id_labels, id_logits, id_feats, ood_labels, ood_logits, ood_feats, train_logits, train_feats, tpr_th, prec_th)
            # elif metric == 'dml':
            #     ood_scores['dml'] = dml(id_labels, id_feats, ood_labels, ood_feats, fc_weight, tpr_th, prec_th)
            # elif metric == 'dmlp':
            #     ood_scores['dmlp'] = dmlp(id_labels, id_logits, id_feats, ood_labels, ood_logits, ood_feats, fc_weight, prototype_cls, tpr_th, prec_th)
            # elif metric == 'cls':
            #     ood_scores['cls'] = prot(id_labels, [id_logits], ood_labels, [ood_logits], [prototype_cls], tpr_th, prec_th)
            # elif metric == 'att':
            #     ood_scores['att'] = prot(id_labels, [id_logits], ood_labels, [ood_logits], [prototype_cls], tpr_th, prec_th)
            # elif metric == 'merge':
            #     ood_scores['merge'] = prot(id_labels, [id_logits], ood_labels, [ood_logits], [prototype_cls], tpr_th, prec_th)
            # elif metric == 'attr':
            #     ood_scores['attr'] = prot(id_labels, [id_attrs], ood_labels, [ood_attrs], [prototype_att], tpr_th, prec_th)
            # else:
            #     raise ValueError(f"Unknown metric: {metric}")

        return ood_scores

    @staticmethod
    def inference(
        metrics,
        logits,
        feat,
        train_logits,
        train_feat,
        labels,
        fc_weight,
        prototype,
        logits_att=None,
        prototype_att=None,
        global_logits=None,
        local_logits=None,
    ) -> dict:
        conf = {}
        for metric in metrics:
            if metric == "mls":
                conf["mls"] = mls_inf(logits)
            elif metric == "msp":
                conf["msp"] = msp_inf(logits)
            elif metric == "mcm":
                conf["mcm"] = mcm_inf(logits)
            elif metric == "glmcm":
                conf["glmcm"] = glmcm_inf(global_logits, local_logits)
            elif metric == "dpm":
                conf["dpm"] = dpm_inf(logits, train_logits)
            # elif metric == 'neglabel':
            #     conf['neglabel'] = neglabel_inf(positive_logits, negative_logits)
            elif metric == "energy":
                conf["energy"] = energy_inf(logits)
            elif metric == "vim":
                conf["vim"] = vim_inf(logits, feat, train_logits, train_feat)
            elif metric == "dml":
                conf["dml"] = dml_inf(feat, fc_weight)
            elif metric == "dmlp":
                conf["dmlp"] = dmlp_inf(logits, feat, fc_weight, prototype)
            elif metric == "klm":
                conf["klm"] = klm_inf(logits, train_logits)
            # elif metric == 'she':
            #     conf['she'] = she_inf(logits, feat, train_feat)
            # elif metric == 'relation':
            #     conf['relation'] = relation_inf(logits, feat, train_logits, train_feat)

            elif metric == "cls":
                conf["cls"] = prot_inf([logits], [prototype])
            elif metric == "att":
                conf["att"] = prot_inf([logits], [prototype])
            elif metric == "merge":
                conf["merge"] = prot_inf([logits], [prototype])
            elif metric == "attr":
                conf["attr"] = prot_inf([logits_att], [prototype_att])
            else:
                raise ValueError(f"Unknown metric: {metric}")

        return conf
