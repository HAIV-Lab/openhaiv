"""
Register
version: 0.0.1
update: 2023.12.13
"""
import warnings


class Register(dict):
    """Register class, used to register functions or classes.

    Args:
        dict (dict): dict to store registered targets.

    Examples:
        >>> register = Register()
        >>> @register
        >>> def foo():
        >>>     return 'foo'
        >>> @register
        >>> def bar():
        >>>     return 'bar'

        >>> print(register)
        {'foo': <function foo at 0x7f9b1c0e0d30>, 'bar': <function bar at 0x7f9b1c0e0e18>}
        >>> print(register['foo'])
        <function foo at 0x7f9b1c0e0d30>
        >>> print(register['bar'])
        <function bar at 0x7f9b1c0e0e18>

        >>> print('foo' in register)
        True
        >>> print('bar' in register)
        True
        >>> print('foobar' in register)
        False

        >>> print(register.keys())
        dict_keys(['foo', 'bar'])
        >>> print(register.values())
        dict_values([<function foo at 0x7f9b1c0e0d30>, <function bar at 0x7f9b1c0e0e18>])
        >>> print(register.items())
        dict_items([('foo', <function foo at 0x7f9b1c0e0d30>), ('bar', <function bar at 0x7f9b1c0e0e18>)])
        >>> print(len(register))
        2

        >>> print(register['foo']())
        foo
        >>> print(register['bar']())
        bar
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self._dict = dict(*args, **kwargs)

    def register_callable(self, target):
        """Register a target.
        Args:
            target (Callable): target to be registered.
        """
        assert callable(target), f"Target {target} is not callable."
        key = target.__name__.lower()
        value = target
        if key in self._dict:
            warnings.warn(f"Target {key} is already registered.")
        self[key] = value

    def register_dict(self, target):
        """Register a dict.
        Args:
            target (dict): dict to be registered.
        """
        assert isinstance(target, dict), f"Target {target} is not a dict."
        for key, value in target.items():
            assert callable(value), f"Target {value} is not callable."
            key = key.lower()
            if key in self._dict:
                warnings.warn(f"Target {key} is already registered.")
            self[key] = value

    def register(self, target):
        """Register a target.
        Args:
            target (Callable or dict): target to be registered.
        """
        if callable(target):
            self.register_callable(target)
        elif isinstance(target, dict):
            self.register_dict(target)
        else:
            raise TypeError(f"Target {target} is not callable or dict.")
    
    def __call__(self, target):
        return self.register(target)
    
    def __setitem__(self, key, value):
        self._dict[key] = value

    def __getitem__(self, key):
        return self._dict[key]

    def __contains__(self, key):
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
