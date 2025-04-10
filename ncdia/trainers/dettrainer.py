import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from ncdia.utils import TRAINERS
from ncdia.dataloader import build_dataloader
from ncdia.algorithms.ood import AutoOOD
from .pretrainer import PreTrainer
from .hooks import QuantifyHook
from ncdia.dataloader import BMF_OOD
import PIL


@TRAINERS.register
class DetTrainer(PreTrainer):
    """Pipeline for OOD detection, 
       including model training and out-of-distribution data detection.

    Args:
        cfg (dict, optional): Configuration for trainer.
        model (nn.Module): Model to be trained.
        max_epochs (int, optional): Maximum number of training epochs.
        gather_train_stats (bool): Whether to gather training statistics.
        gather_test_stats (bool): Whether to gather testing statistics.
        eval_loader (DataLoader, optional): DataLoader for OOD evaluation.
        verbose (bool, optional): Whether to print training information.

    Attributes:
        eval_loader (DataLoader): DataLoader for OOD evaluation.
        quantify_hook (QuantifyHook): Hook for dataset statistics collection.
        max_epochs (int): Maximum number of training epochs.
        verbose (bool): Whether to print training information.

    """
    def __init__(
            self,
            cfg: dict | None = None,
            model: nn.Module = None,
            max_epochs: int = 1,
            gather_train_stats: bool = False,
            gather_test_stats: bool = False,
            eval_loader: DataLoader | dict | None = None,
            verbose: bool = False,
            **kwargs
    ) -> None:
        self.quantify_hook = QuantifyHook(
            gather_train_stats=gather_train_stats,
            gather_test_stats=gather_test_stats,
            verbose=verbose
        )
        super(DetTrainer, self).__init__(
            cfg=cfg,
            model=model,
            max_epochs=max_epochs,
            custom_hooks=[self.quantify_hook],
            **kwargs
        )

        self.eval_loader_tmp = eval_loader
        self._eval_loader = {}
        if 'evalloader' in self._cfg:
            self._eval_loader.update(dict(self._cfg['evalloader']))
        if isinstance(eval_loader, dict):
            self._eval_loader.update(eval_loader)
        elif isinstance(eval_loader, DataLoader):
            self._eval_loader = eval_loader

        self.kwargs = kwargs
        self.verbose = verbose
        # image = PIL.Image.open("/new_data/datasets/OES/sub-dataset1-RGB-domain1/ID/test/Abies alba/Abies_alba_2_370_WEFL_NLF.jpg")
        # # 定义预处理
        # preprocess = transforms.Compose([
        #     transforms.Resize((224, 224)),  # 调整图像大小
        #     transforms.ToTensor(),          # 转换为张量
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
        # ])

        # # 对 PIL 图像进行预处理
        # image_tensor = preprocess(image).unsqueeze(0)  # 添加 batch 维度
        # # 将图像张量移动到设备上
        # image_tensor = image_tensor.to(self.device)
        # # 调用模型
        # logits = self.model(image_tensor)
        # print(f"Logits: {logits}")

    # @property
    # def eval_loader(self) -> DataLoader:
    #     """DataLoader: DataLoader for OOD evaluation."""
    #     if not self._eval_loader:
    #         return None
    #     if isinstance(self._eval_loader, dict):
    #         self._eval_loader, self._eval_dataset_kwargs, \
    #             self._eval_loader_kwargs = build_dataloader(self._eval_loader)
    #     return self._eval_loader

    @property
    def eval_loader(self) -> DataLoader:
        """DataLoader: DataLoader for OOD evaluation."""
        self._eval_loader = {}
        if 'evalloader' in self._cfg:
            self._eval_loader.update(dict(self._cfg['evalloader']))
        if isinstance(self.eval_loader_tmp, dict):
            self._eval_loader.update(self.eval_loader_tmp)
        elif isinstance(self.eval_loader_tmp, DataLoader):
            self._eval_loader = self.eval_loader_tmp
        print(f"Eval loader: {self._eval_loader}")
        return self._eval_loader
    
    @property
    def train_stats(self) -> dict:
        """Get training stats, including features, logits, labels, and prototypes."""
        if "_train_stats" not in self.__dict__:
            return None
        return self._train_stats
    
    @property
    def test_stats(self) -> dict:
        """Get testing stats, including features, logits, labeld, and prototypes."""
        if "_test_stats" not in self.__dict__:
            return None
        return self._test_stats
    
    def train(self):
        """Training and evaluation for out-of-distribution (OOD) detection.
        Firstly, train a model, and then evaluate the model on the OOD dataset.

        Returns:
            model (nn.Module): Trained model.
        """
        super(DetTrainer, self).train()

        # If the configuration of eval_loader is provided,
        # then evaluate the model on the eval_loader.
        # If not, trainer only train the model but not run OOD detection.
        # Furthermore, OOD detection can also be run after training 
        # by calling `trainer.evaluate(evalloader=DataLoader)`.
        if self.eval_loader:
            self.evaluate()

        return self.model

    def evaluate(
            self,
            metrics: list = ['msp'],
            evalloader: DataLoader = None,
            tpr_th: float = 0.95,
            prec_th: float = None
    ) -> dict:
        """Evaluate dataset.

        Args:
            metrics (list, optional): list of OOD detection methods to evaluate.
            evalloader (DataLoader, optional): DataLoader for OOD evaluation.
            tpr_th (float, optional): True positive rate threshold. Defaults to 0.95.
            prec_th (float, optional): Precision threshold. Defaults to None.

        Returns:
            dict: OOD scores, keys are the names of the OOD detection methods,
                values are the OOD scores and search threshold.
        """
        # train_stats = self.train_stats
        # test_stats = self.test_stats

        # eval_stats = self.quantify_hook.gather_stats(
        #     model=self.model,
        #     dataloader=evalloader if evalloader else self.eval_loader,
        #     device=self.device,
        #     verbose=self.verbose
        # )

        # scores = self.algorithm.eval(
        #     id_gt=test_stats['labels'],
        #     id_logits=test_stats['logits'],
        #     id_feat=test_stats['features'],
        #     ood_logits=eval_stats['logits'],
        #     ood_feat=eval_stats['features'],
        #     train_logits=train_stats['logits'],
        #     train_feat=train_stats['features'],
        #     tpr_th=tpr_th,
        #     prec_th=prec_th,
        #     hyparameters=self.algorithm.hyparameters if hasattr(self.algorithm, 'hyparameters') else None
        # )

        # return scores
        self.model.eval()
        for setting_name, id_setting in self._eval_loader.items():
            print('*************evaluate {} setting*************'.format(setting_name))

            # for dataset_name, data_cfg in id_setting.items():
            if True:
                dataset_name = setting_name
                evalset = BMF_OOD(
                    # root = data_cfg['root'],
                    # split = data_cfg['split'],
                    root = id_setting['root'],
                    split = id_setting['split'],
                    subset_labels = None,
                    subset_file = None,
                    transform = None)
                evalloader = DataLoader(
                    evalset,
                    batch_size=32,
                    shuffle=False,
                    num_workers=8,
                )
                if dataset_name == 'dataset':
                    test_stats = self.quantify_hook.gather_stats(
                        model=self.model,
                        dataloader=evalloader,
                        device=self.device,
                        # id_acc = True,
                        verbose=self.verbose
                    )
                else:
                    print('*************evaluate {} Datasets*************'.format(dataset_name))
                    eval_stats = self.quantify_hook.gather_stats(
                        model=self.model,
                        dataloader=evalloader,
                        device=self.device,
                        # id_acc=False,
                        verbose=self.verbose
                    )
        scores = self.algorithm.eval(
            id_gt=test_stats['labels'],
            id_logits=test_stats['logits'],
            id_feat=test_stats['features'],
            ood_logits=eval_stats['logits'],
            ood_feat=eval_stats['features'],
            # train_logits=train_logits,
            # train_feat=train_feat,
            # train_gt=train_gt,
            # tpr_th=tpr_th,
            # prec_th=prec_th,
            hyparameters=self.algorithm.hyparameters if hasattr(self.algorithm, 'hyparameters') else None
        )
        # print(f"scores: {scores}")
        # print('FPR95:',scores[0][0]*100)
        # print('AUROC:',scores[0][1]*100)

        exit(0)
    
        
    def detect(
            self,
            metrics: list = ['msp'],
            evalloader: DataLoader = None,
    ):
        """Detect OOD data, and return evaluated confidence.

        Args:
            metrics (list, optional): list of OOD detection methods to evaluate.
            evalloader (DataLoader, optional): DataLoader for OOD evaluation.

        Returns:
            dict: OOD confidence, keys are the names of the OOD detection methods,
                values are the OOD confidence.
        """
        train_stats = self.train_stats
        test_stats = self.test_stats

        eval_stats = self.quantify_hook.gather_stats(
            model=self.model,
            dataloader=evalloader if evalloader else self.eval_loader,
            device=self.device,
            verbose=self.verbose
        )

        confidence = AutoOOD().inference(
            metrics=metrics,
            logits=eval_stats['logits'],
            feat=eval_stats['features'],
            train_logits=train_stats['logits'],
            train_feat=train_stats['features'],
            fc_weight=self.model.fc.weight.clone().detach().cpu(),
            prototype=train_stats['prototypes']
        )

        return confidence
