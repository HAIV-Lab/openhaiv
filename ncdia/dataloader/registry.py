class Register(object):
    """base register
    
    Args: 
        name (str): name of the register
    
    Examples:
        >>> reg = Register("reg")
        >>> @reg.register_module()
        >>> class xxx(xxxx):
                def ...
    """
    def __init__(self, name: str) -> None:
        self.module_dict = {}
        self.name = name
    
    def __repr__(self) -> str:
        pstr = "The items registered in {}: \n".format(self.name)
        for key, _ in self.modules().items():
            pstr += (key + '\n')
        return pstr
    
    def register_module(self):
        def _register(target):
            assert hasattr(target, '__name__')
            name = target.__name__.lower()
            if name not in self.modules().keys():
                self.module_dict[name] = target
                print("Class {} is registered!".format(name.upper()))
            return target

        return _register
    
    def modules(self):
        return self.module_dict


class DatasetRegister(Register):
    """dataset register
    
    Args: 
        name (str): name of the dataset register
    
    Examples:
        >>> DATASETS = DatasetRegister("reg")
        >>> @DATASETS.register_module()
        >>> class xxx(iData):
                def ...
    """
    def __init__(self, name="Dataset Register") -> None:
        super().__init__(name)
    
    def register_module(self):
        def _register(target):
            assert hasattr(target, '__name__')
            name = target.__name__.lower()
            if name not in self.modules().keys():
                assert issubclass(target, iData)
                self.module_dict[name] = target
                print("Class {} is registered!".format(name.upper()))
            return target

        return _register


class iData(object):
    """a dataset template for DatasetRegister

    Args: 
        path (str): path to the dataset (if the dataset is saved locally)
    """
    def __init__(self, path: str = '') -> None:
        self.class_order = None
        self.path = path

        self.train_trsf = []
        self.test_trsf = []
        self.ncd_trsf = []
        self.common_trsf = []

        self.train_data = None
        self.train_targets = None
        self.test_data = None
        self.test_targets = None
        self.ncd_data = None
        self.ncd_targets = None
