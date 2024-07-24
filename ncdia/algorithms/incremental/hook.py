from ncdia.trainers.hooks import Hook


class FACTHook(Hook):
    
    def __init__(self) -> None:
        super().__init__()
    
    def after_train(self, trainer) -> None:
        algorithm = trainer.algorithm
        algorithm.replace_fc()