import os
import torch
import numpy as np
from torch import optim
from requests import session
from tqdm import tqdm

from ncdia.utils import ALGORITHMS
from ncdia.algorithms.base import BaseAlg
from ncdia.utils.losses import AngularPenaltySMLoss
from ncdia.utils.metrics import accuracy, per_class_accuracy
from ncdia.utils import HOOKS
from ncdia.trainers.hooks import AlgHook
from ncdia.trainers.optims import build_optimizer, build_scheduler
from ncdia.models.net.foster_net import FOSTERNet


@HOOKS.register
class FosterHook(AlgHook):
    def __init__(self) -> None:
        super().__init__()

    def before_train(self, trainer) -> None:
        if trainer.session > 0:
            total_class = (
                trainer.cfg.CIL.base_classes + trainer.session * trainer.cfg.CIL.way
            )
            trainer.model.update_fc(total_class)
            trainer._optimizer = optim.SGD(
                filter(lambda p: p.requires_grad, trainer.model.parameters()),
                lr=trainer.cfg.optimizer.lr,
                weight_decay=trainer.cfg.optimizer.weight_decay,
            )
            trainer._scheduler = optim.lr_scheduler.ConstantLR(
                optimizer=trainer._optimizer
            )

    def after_train(self, trainer) -> None:
        algorithm = trainer.algorithm
        filename = "task_" + str(trainer.session) + ".pth"
        trainer.save_ckpt(os.path.join(trainer.work_dir, filename))
        total_class = (
            trainer.cfg.CIL.base_classes + trainer.session * trainer.cfg.CIL.way
        )
        trainer.buffer["student_network"] = FOSTERNet(
            trainer.cfg.model.network,
            trainer.cfg.CIL.base_classes,
            trainer.cfg.CIL.num_classes,
            trainer.cfg.CIL.att_classes,
            trainer.cfg.model.net_alice,
            total_classes=total_class,
            pretrained=False,
        )
        if trainer.session > 0:
            algorithm.init_student()
            max_epoch = 15
            optimizer = optim.SGD(
                filter(
                    lambda p: p.requires_grad,
                    trainer.buffer["student_network"].parameters(),
                ),
                lr=0.002,
                momentum=0.9,
            )
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer, T_max=max_epoch
            )
            for epoch in range(max_epoch):
                trainer.epoch = epoch
                for batch_idx, batch in enumerate(
                    tqdm(trainer.train_loader, desc="Training Student")
                ):
                    data, label, attribute, imgpath = trainer.batch_parser(batch)
                    algorithm.train_step_student(
                        trainer, data, label, attribute, imgpath, optimizer
                    )
                for batch_idx, batch in enumerate(
                    tqdm(trainer.val_loader, desc="Validating Student")
                ):
                    data, label, attribute, imgpath = trainer.batch_parser(batch)
                    algorithm.val_step_student(trainer, data, label, attribute, imgpath)
                scheduler.step()
            trainer.model.replace(trainer.buffer["student_network"])


@ALGORITHMS.register
class Foster(BaseAlg):
    def __init__(self, trainer) -> None:
        super().__init__(trainer)
        self.args = trainer.cfg
        self.trainer = trainer

        self._network = None
        self._student_network = None
        self.transform = None
        self.beta1 = 0.96
        self.beta2 = 0.97
        self.T = 2
        self.class_num_list = [0] * self.args.CIL.num_classes
        self.per_class_weights = None
        self.weight_align_teacher = False
        self.weight_align_student = False
        self.loss = torch.nn.CrossEntropyLoss().cuda()
        self.hook = FosterHook()
        trainer.register_hook(self.hook)

        session = trainer.session

    def train_step(self, trainer, data, label, attribute, imgpath):
        """
        base train for foster method
        Args:
            data: data in batch
            label: label in batch
            attribute: attribute in batch
            imgpath: imgpath in batch
        """

        session = self.trainer.session
        if session == 0:
            known_class = self.args.CIL.base_classes
        else:
            known_class = self.args.CIL.base_classes + (session - 1) * self.args.CIL.way
        total_class = self.args.CIL.base_classes + session * self.args.CIL.way
        self._network = trainer.model
        self._network.train()

        data = data.cuda()
        labels = label.cuda()
        if session == 0:
            logits, fe_logits, old_logits = self._network(data)
            logits_ = logits[:, :total_class]
            acc = accuracy(logits_, labels)[0]
            per_acc = str(per_class_accuracy(logits_, labels))
            loss = self.loss(logits_, labels)
            loss.backward()

            ret = {"loss": loss, "acc": acc, "per_class_acc": per_acc}
        else:
            self.update_class_weights(labels)
            ret = self._feature_boosting(data, labels)
            if self.weight_align_teacher:
                self._network.weight_align(known_class)
        return ret

    def train_step_student(self, trainer, data, label, attribute, imgpath, optimizer):
        session = self.trainer.session
        if session == 0:
            known_class = self.args.CIL.base_classes
        else:
            known_class = self.args.CIL.base_classes + (session - 1) * self.args.CIL.way
        self._network = trainer.model
        self._network.eval()
        self._student_network.train()

        data = data.cuda()
        labels = label.cuda()
        if session > 0:
            ret = self._feature_compression(data, labels, optimizer)
            if self.weight_align_student:
                self._student_network.weight_align(known_class)
        else:
            ret = None
        return ret

    def val_step(self, trainer, data, label, *args, **kwargs):
        """Validation step for foster.

        Args:
            trainer (object): Trainer object.
            data (torch.Tensor): Input data.
            label (torch.Tensor): Label data.
            args (tuple): Additional arguments.
            kwargs (dict): Additional keyword arguments.

        Returns:
            results (dict): Validation results. Contains the following:
                - "feature" (numpy.array): features in a batch
                - "logits" (numpy.array): logits in a batch
                - "predicts" (numpy.array): predicts in a batch
                - "confidence" (numpy.array): confidence in a batch
                - "label" (numpy.array): labels in a batch
                - "loss": Loss value.
                - "acc": Accuracy value.
        """
        session = self.trainer.session
        test_class = self.args.CIL.base_classes + session * self.args.CIL.way
        self._network = trainer.model
        self._network.eval()
        data = data.cuda()
        labels = label.cuda()
        logits, fe_logits, old_logits = self._network(data)
        logits_ = logits[:, :test_class]
        acc = accuracy(logits_, labels)[0]
        loss = self.loss(logits_, labels)
        # per_acc = str(per_class_accuracy(logits_, labels))

        ret = {}
        ret["loss"] = loss.item()
        ret["acc"] = acc.item()
        # ret['per_class_acc'] = per_acc

        return ret

    def val_step_student(self, trainer, data, label, *args, **kwargs):
        session = self.trainer.session
        test_class = self.args.CIL.base_classes + session * self.args.CIL.way
        self._network = trainer.buffer["student_network"]
        self._network.eval()
        data = data.cuda()
        labels = label.cuda()
        logits, fe_logits, old_logits = self._network(data)
        logits_ = logits[:, :test_class]
        acc = accuracy(logits_, labels)[0]
        loss = self.loss(logits_, labels)

        ret = {}
        ret["loss"] = loss.item()
        ret["acc"] = acc.item()

        return ret

    def test_step(self, trainer, data, label, *args, **kwargs):
        return self.val_step(trainer, data, label, *args, **kwargs)

    def get_net(self):
        return self._network

    def _KD_loss(self, pred, soft, T):
        pred = torch.log_softmax(pred / T, dim=1)
        soft = torch.softmax(soft / T, dim=1)
        return -1 * torch.mul(soft, pred).sum() / pred.shape[0]

    def BKD(self, pred, soft, T):
        pred = torch.log_softmax(pred / T, dim=1)
        soft = torch.softmax(soft / T, dim=1)
        soft = soft * self.per_class_weights
        soft = soft / soft.sum(1)[:, None]
        return -1 * torch.mul(soft, pred).sum() / pred.shape[0]

    def _feature_boosting(self, data, labels):
        session = self.trainer.session
        if session == 0:
            known_class = self.args.CIL.base_classes
        else:
            known_class = self.args.CIL.base_classes + (session - 1) * self.args.CIL.way
        total_class = self.args.CIL.base_classes + session * self.args.CIL.way

        logits, fe_logits, old_logits = self._network(data)
        logits_ = logits[:, :total_class]
        fe_logits_ = fe_logits[:, :total_class]
        loss_clf = self.loss(logits_, labels)
        loss_fe = self.loss(fe_logits_, labels)
        loss_kd = self._KD_loss(logits[:, :known_class], old_logits, self.T)
        loss = loss_clf + loss_fe + loss_kd
        acc = accuracy(logits_, labels)[0]
        per_acc = str(per_class_accuracy(logits_, labels))
        loss.backward()
        ret = {"loss": loss, "acc": acc, "per_class_acc": per_acc}

        return ret

    def update_class_weights(self, labels):
        session = self.trainer.session
        if session == 0:
            known_class = self.args.CIL.base_classes
        else:
            known_class = self.args.CIL.base_classes + (session - 1) * self.args.CIL.way
        total_class = self.args.CIL.base_classes + session * self.args.CIL.way
        class_num_new = torch.bincount(
            labels, minlength=self.args.CIL.num_classes
        ).data.tolist()
        self.class_num_list = [
            x + y for x, y in zip(class_num_new, self.class_num_list)
        ]
        effective_num = 1.0 - np.power(self.beta2, self.class_num_list[:total_class])
        per_cls_weights = (1.0 - self.beta2) / (np.array(effective_num) + 1e-6)
        per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * total_class
        self.per_class_weights = torch.FloatTensor(per_cls_weights).cuda()

    def init_student(self):
        session = self.trainer.session
        self._network = self.trainer.model
        if hasattr(self.trainer.buffer["student_network"], "module"):
            self._student_network = self.trainer.buffer["student_network"].module
        else:
            self._student_network = self.trainer.buffer["student_network"]
        self._student_network.cuda()
        self._student_network.convnets[0].load_state_dict(
            self._network.convnets[0].state_dict()
        )
        self._student_network.copy_fc(self._network.old_fc)

    def _feature_compression(self, data, labels, optimizer):
        session = self.trainer.session
        total_class = self.args.CIL.base_classes + session * self.args.CIL.way
        dark_logits, fe_logits, old_logits = self._student_network(data)
        dark_logits_ = dark_logits[:, :total_class]
        with torch.no_grad():
            logits, fe_logits, old_logits = self._network(data)
            logits_ = logits[:, :total_class]
        loss_dark = self.BKD(dark_logits_, logits_, self.T)
        loss = loss_dark
        acc = accuracy(dark_logits_, labels)[0]
        per_acc = str(per_class_accuracy(dark_logits_, labels))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ret = {"loss": loss, "acc": acc, "per_class_acc": per_acc}
        return ret
