from torch.utils.data import DataLoader

from ncdia.utils import ALGORITHMS
from ncdia.algorithms.base import BaseAlg
from ncdia.utils import HOOKS
from ncdia.trainers.hooks import AlgHook
from ncdia.dataloader import MergedDataset


@HOOKS.register
class ExpHook(AlgHook):
    def __init__(self) -> None:
        super().__init__()

    def before_train(self, trainer) -> None:
        """
        在进入for epoch in range(max_epochs)循环之前，对训练数据集进行处理。
        """
        # 初始化trainer的数据加载器
        trainer.train_loader
        
        # 获取IncTrainer上一个session保存的历史数据集
        _hist_trainset = MergedDataset([trainer.hist_trainset], replace_transform=True)
        
        # 将历史数据集与当前数据集合并
        _hist_trainset.merge([trainer.train_loader.dataset], replace_transform=True)
        
        # 重新设置IncTrainer的数据加载器
        trainer._train_loader = DataLoader(_hist_trainset, **trainer._train_loader_kwargs)

    def after_train(self, trainer) -> None:
        """
        在训练结束后，将当前session中需要保存的数据保存到hist_trainset中。
        """
        # Example：保存当前session的所有训练数据。
        # 由于当前session的训练数据集已经融合和历史数据，
        # 所以只需要用当前session的数据覆盖hist_trainset即可。

        # 将inplace设置为True，直接替换hist_trainset
        trainer.update_hist_trainset(
            trainer.train_loader.dataset,
            replace_transform=True,
            inplace=True
        )
    
    def before_val(self, trainer) -> None:
        """
        在进入for epoch in range(max_epochs)循环之前，对验证数据集进行处理。
        """
        trainer.val_loader
        _hist_valset = MergedDataset([trainer.hist_valset], replace_transform=True)
        _hist_valset.merge([trainer.val_loader.dataset], replace_transform=True)
        trainer._val_loader = DataLoader(_hist_valset, **trainer._val_loader_kwargs)

    def after_val(self, trainer) -> None:
        """
        在验证结束后，将当前session中需要保存的数据保存到hist_valset中。
        """
        trainer.update_hist_valset(
            trainer.val_loader.dataset,
            replace_transform=True,
            inplace=True
        )

    def before_test(self, trainer) -> None:
        """
        在进入for epoch in range(max_epochs)循环之前，对测试数据集进行处理。
        """
        trainer.test_loader
        _hist_testset = MergedDataset([trainer.hist_testset], replace_transform=True)
        _hist_testset.merge([trainer.test_loader.dataset], replace_transform=True)
        trainer._test_loader = DataLoader(_hist_testset, **trainer._test_loader_kwargs)

    def after_test(self, trainer) -> None:
        """
        在测试结束后，将当前session中需要保存的数据保存到hist_testset中。
        """
        trainer.update_hist_testset(
            trainer.test_loader.dataset,
            replace_transform=True,
            inplace=True
        )


@ALGORITHMS.register
class Example(BaseAlg):
    def __init__(self, trainer) -> None:
        super().__init__(trainer)
        self.trainer = trainer

        hook = ExpHook()
        trainer.register_hook(hook)
