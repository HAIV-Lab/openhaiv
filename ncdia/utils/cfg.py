import os
import yaml
from copy import deepcopy

_cfg = None


def get_cfg():
    """Get global config.

    Returns:
        Configs (Configs): global config
    """
    global _cfg
    return _cfg


def parse_value(value: list | dict | bool | int | float | None):
    """Parse value from yaml files to python type.

    Args:
        value (str|list|dict|bool|int|float|None): value to be parsed

    Returns:
        value (str|list|dict|bool|int|float|None): parsed value

    Examples:
        >>> parse_value('1')
        1
        >>> parse_value('1.0')
        1.0
        >>> parse_value('True')
        True
        >>> parse_value('None')
        None
        >>> parse_value('abc')
        'abc'
        >>> parse_value('abc.def')
        'abc.def'
        >>> parse_value(['1', '2', '3'])
        [1, 2, 3]
        >>> parse_value({'a': '1', 'b': '2'})
        {'a': 1, 'b': 2}
        >>> parse_value({'a': '1', 'b': ['2', '3']})
        {'a': 1, 'b': [2, 3]}
        >>> parse_value({'a': '1', 'b': {'c': '2', 'd': '3'}})
        {'a': 1, 'b': {'c': 2, 'd': 3}}
    """
    if isinstance(value, list):
        return [parse_value(v) for v in value]
    if isinstance(value, dict):
        return {k: parse_value(v) for k, v in value.items()}
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    if isinstance(value, str):
        if value.lower() in ["true", "false"]:
            return value.lower() == "true"
        try:
            return int(value)
        except ValueError:
            pass
        try:
            return float(value)
        except ValueError:
            pass
        return value
    return value


def set_value(obj: dict, key: str, value):
    """Set value to dict by key.

    Args:
        obj (dict): dict to be set
        key (str): key to be set, support nested key by '.' (e.g. 'a.b.c')
        value (any): value to be set

    Examples:
        >>> cfg = {}
        >>> set_value(cfg, 'a', 1)
        >>> cfg
        {'a': 1}
        >>> set_value(cfg, 'b.c', 2)
        >>> cfg
        {'a': 1, 'b': {'c': 2}}
        >>> set_value(cfg, 'b.d', 3)
        >>> cfg
        {'a': 1, 'b': {'c': 2, 'd': 3}}
    """
    keys = str(key).split(".")
    if len(keys) == 1:
        obj[keys[0]] = value
    else:
        if keys[0] not in obj:
            obj[keys[0]] = {}
        set_value(obj[keys[0]], ".".join(keys[1:]), value)


class Configs:
    """Configs class.

    Args:
        obj (dict, optional): Dict to be parsed. Defaults to None.
        name (str, optional): Name of the config. Defaults to 'config'.

    Examples:
        >>> cfg = Configs({'a': 1, 'b': {'c': 2}})
        >>> cfg.a
        1
        >>> cfg.b.c
        2
        >>> cfg['a']
        1
        >>> cfg['b.c']
        2

        >>> cfg = Configs()
        >>> cfg.merge_from_dict({'a': 1, 'b': {'c': 2}})
        >>> cfg.a
        1
        >>> cfg.b.c
        2

        >>> cfg = Configs()
        >>> cfg.merge_from_yaml('tests/data/config.yaml')
        >>> cfg.a
        1
        >>> cfg.b.c
        2

        >>> cfg = Configs()
        >>> cfg.merge_from_list(['a=1', 'b.c=2'])
        >>> cfg.a
        1
        >>> cfg.b.c
        2
    """

    def __init__(self, obj=None, name="config"):
        self.cfg = {}
        self._name = name

        if obj is not None:
            for k, v in obj.items():
                v = parse_value(v)
                self.cfg[k] = v

                if isinstance(v, dict):
                    v = Configs(v, name=k)
                setattr(self, k, v)

    def merge_from_yaml(self, cfg_file: str):
        """Merge configs from yaml file.

        Args:
            cfg_file (str): path to yaml file
        """
        assert cfg_file.split(".")[-1] in [
            "yaml",
            "yml",
        ], "Only yaml files are supported"
        cfg_file = os.path.abspath(os.path.expanduser(cfg_file))

        with open(cfg_file, "r") as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
            if cfg is None:
                cfg = {}

        if "_base_" in cfg:
            if not isinstance(cfg["_base_"], list):
                cfg["_base_"] = [cfg["_base_"]]

            for base_cfg_file in cfg["_base_"]:
                self.merge_from_yaml(base_cfg_file)

        self.merge_from_dict(cfg)

    def merge_from_dict(self, args: dict, cfg: dict = None):
        """Merge configs from dict.

        Args:
            args (dict): dict to be merged
            cfg (dict, optional): dict to be merged to. Defaults to None.
        """
        if cfg is None:
            cfg = self.cfg

        for k, v in args.items():
            if k in cfg and isinstance(v, dict) and isinstance(cfg[k], dict):
                if "_delete_" in v and v["_delete_"]:
                    cfg.pop(k)
                elif "_replace_" in v and v["_replace_"]:
                    v.pop("_replace_")
                    cfg[k] = v
                else:
                    self.merge_from_dict(v, cfg[k])
            else:
                if isinstance(v, dict) and "_delete_" in v:
                    if v["_delete_"]:
                        continue
                    else:
                        v.pop("_delete_")
                elif isinstance(v, dict) and "_replace_" in v:
                    v.pop("_replace_")
                cfg[k] = v

    def merge_from_list(self, args: list):
        """Merge configs from list.

        Args:
            args (list): list to be merged
        """
        args_dict = {}
        for arg in args:
            try:
                k, v = arg.split("=")
            except ValueError:
                raise ValueError("Argument must be in format k=v")
            set_value(args_dict, k.strip(), v.strip())

        self.merge_from_dict(args_dict)

    def merge_from_config(self, cfg):
        """Merge configs from another Configs.

        Args:
            cfg (Configs | dict): Configs to be merged
        """
        if isinstance(cfg, Configs):
            cfg = cfg.cfg

        if "_base_" in cfg:
            if not isinstance(cfg["_base_"], list):
                cfg["_base_"] = [cfg["_base_"]]

            for base_cfg_file in cfg["_base_"]:
                self.merge_from_yaml(base_cfg_file)

        self.merge_from_dict(cfg)

    def freeze(self):
        """Freeze configs, construct a tree of Configs,
        and set it to global variable _cfg,
        which can be accessed by
        ```
        >>> from utils.config import get_cfg
        >>> cfg = get_cfg()
        ```
        """
        cfg = self.cfg
        self.__init__(cfg)

        global _cfg
        _cfg = self

    def __getitem__(self, key):
        key_path = str(key).split(".")
        _key = key_path[0]

        if _key in self.__dict__:
            _value = getattr(self, _key)

            if len(key_path) == 1:
                return _value
            elif isinstance(_value, Configs):
                return _value[".".join(key_path[1:])]
            else:
                raise KeyError('Key "{}" not found in config'.format(key))

        else:
            raise KeyError('Key "{}" not found in config'.format(key))

    def __str__(self, indent=0):
        content = ""

        for _key in self.__dict__:
            _value = getattr(self, _key)

            if _key in ["cfg", "_name"]:
                continue
            elif not isinstance(_value, Configs):
                content += " " * indent + f"{_key}: {_value}\n"
            else:
                content += " " * indent + f"{_key}:\n"
                content += _value.__str__(indent + 2)

        return content

    def __getattribute__(self, name: str):
        try:
            return object.__getattribute__(self, name)
        except AttributeError:
            return None

    def __contains__(self, key):
        return key in self.cfg

    def __len__(self):
        return len(self.cfg)

    def keys(self):
        return self.cfg.keys()

    def values(self):
        return self.cfg.values()

    def items(self):
        return self.cfg.items()

    def __deepcopy__(self, memo):
        new_obj = self.__class__()
        for key, value in self.__dict__.items():
            setattr(new_obj, key, deepcopy(value, memo))
        return new_obj


def setup_cfg(args, default: dict = {}):
    """Setup config with parsed arguments.

    Args:
        args (argparse.Namespace): parsed arguments
        default (dict): default config

    Returns:
        cfg (Configs): config
    """
    cfg = Configs()

    # Update default config
    default_cfg = {
        "seed": -1,
        "device": "cuda",
        "work_dir": "./results/",
    }
    for key, value in default.items():
        default_cfg[key] = value

    # Set default vaules
    cfg.merge_from_dict(default_cfg)

    # Load config from files
    for cfg_file in args.config_files:
        cfg.merge_from_yaml(cfg_file)

    for opt in args.opts:
        if opt.startswith("_base_"):
            cfg.merge_from_yaml(opt.split("=")[1])
            args.opts.remove(opt)

    # Load config from input arguments
    cfg.merge_from_dict(vars(args))

    # Load config from command line
    cfg.merge_from_list(args.opts)

    cfg.freeze()

    return cfg
