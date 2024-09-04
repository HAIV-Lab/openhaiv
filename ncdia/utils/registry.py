import warnings
from typing import Callable

from .cfg import Configs


class Registry(dict):
    """A registry to map strings to classes or functions.

    Args:
        kwargs (dict): dict to store registered targets.

    Examples:
        >>> REGISTRY = Registry()
        >>> @REGISTRY
        >>> def foo():
        >>>     return 'foo'
        >>> @REGISTRY.register
        >>> def bar():
        >>>     return 'bar'

        >>> print(REGISTRY['foo']())
        foo
        >>> print(REGISTRY['bar']())
        bar

        >>> print(REGISTRY)
        {'foo': <function foo at 0x7f9b1c0e0d30>, 'bar': <function bar at 0x7f9b1c0e0e18>}
        >>> print(REGISTRY['foo'])
        <function foo at 0x7f9b1c0e0d30>
        >>> print(REGISTRY['bar'])
        <function bar at 0x7f9b1c0e0e18>

        >>> print('foo' in REGISTRY)
        True
        >>> print('bar' in REGISTRY)
        True
        >>> print('foobar' in REGISTRY)
        False

        >>> print(REGISTRY.keys())
        dict_keys(['foo', 'bar'])
        >>> print(REGISTRY.values())
        dict_values([<function foo at 0x7f9b1c0e0d30>, <function bar at 0x7f9b1c0e0e18>])
        >>> print(REGISTRY.items())
        dict_items([('foo', <function foo at 0x7f9b1c0e0d30>), ('bar', <function bar at 0x7f9b1c0e0e18>)])
        >>> print(len(REGISTRY))
        2
    """
    def __init__(self, **kwargs):
        super(Registry, self).__init__()
        self._dict = dict(**kwargs)

    def register_callable(self, target: callable):
        """Register a target.
        
        Args:
            target (callable): callable target to be registered.

        Raises:
            TypeError: If target is not callable.
        """
        if not callable(target):
            raise TypeError(f"Target {target} is not callable.")
        
        key = target.__name__.lower()
        value = target
        # if key in self._dict:
        #     warnings.warn(f"Target {key} is already registered.")
        self[key] = value

    def register_dict(self, target: dict):
        """Register a dict.

        Args:
            target (dict): A dict to be registered.
                All its values should be callable.

        Raises:
            TypeError: If target is not a dict.
            TypeError: If any value in target is not callable.
        """
        if not isinstance(target, dict):
            raise TypeError(f"Target {target} is not a dict.")
        
        for key, value in target.items():
            if not callable(value):
                raise TypeError(f"Target {value} is not callable.")
            key = key.lower()
            # if key in self._dict:
            #     warnings.warn(f"Target {key} is already registered.")
            self[key] = value

    def register(self, target: Callable | dict):
        """Register a target.

        Args:
            target (callable | dict): target to be registered.

        Returns:
            target (object): Registered target.

        Raises:
            TypeError: If target is not callable or dict.
        """
        if callable(target):
            self.register_callable(target)
        elif isinstance(target, dict):
            self.register_dict(target)
        else:
            raise TypeError(f"Target {target} is not callable or dict.")
        return target
        
    def build(self, target: dict | Configs, **kwargs):
        """Build a target with configs.

        Args:
            target (dict | Configs): A dict to be built.
                It should have a key 'type' to specify the target type.
                It may have other keys to specify the target configs.
            kwargs (dict): Additional keyword arguments.

        Returns:
            target (object): A built target.

        Raises:
            TypeError: If target is not a dict.
            KeyError: If key 'type' is not found in target.
            KeyError: If target type is not registered.
        """
        if isinstance(target, Configs):
            target = dict(target)

        if not isinstance(target, dict):
            raise TypeError(f"Target {target} is not a dict.")
        else:
            target.update(kwargs)

        if 'type' not in target:
            raise KeyError(f"Key 'type' is not found in target {target}.")
        
        target_type = target['type'].lower()
        if target_type not in self._dict:
            raise KeyError(f"Target type {target_type} is not registered.")
        
        target.pop('type')
        target = self[target_type](**target)
        
        return target
    
    def __call__(self, target):
        return self.register(target)
    
    def __setitem__(self, key, value):
        self._dict[key] = value

    def __getitem__(self, key):
        key = key.lower()
        return self._dict[key]

    def __contains__(self, key):
        key = key.lower()
        return key in self._dict
    
    def __str__(self):
        return str(self._dict)

    def __len__(self):
        return len(self._dict)
    
    def keys(self):
        return self._dict.keys()
    
    def values(self):
        return self._dict.values()
    
    def items(self):
        return self._dict.items()
