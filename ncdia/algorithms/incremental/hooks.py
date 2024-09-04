from ncdia.utils import HOOKS
from ncdia.trainers.hooks import AlgHook


@HOOKS.register
class FACTHook(AlgHook):
    def __init__(self) -> None:
        super().__init__()
    
    def after_train(self, trainer) -> None:
        algorithm = trainer.algorithm
        algorithm.replace_fc()


@HOOKS.register
class AliceHook(AlgHook):
    def __init__(self) -> None:
        super().__init__()
    
    def after_train(self, trainer) -> None:
        algorithm = trainer.algorithm
        algorithm.replace_fc()

@HOOKS.register
class SAVCHook(AlgHook):
    def __init__(self) -> None:
        super().__init__()
    
    def after_train(self, trainer) -> None:
        algorithm = trainer.algorithm
        algorithm.replace_fc()
    
    def before_train(self, trainer) -> None:
        trainer.train_loader.dataset.multi_train = True
