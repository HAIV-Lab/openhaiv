import inspect
import logging
import sys
from collections.abc import Callable
from contextlib import contextmanager
from importlib import import_module
from typing import Any, Dict, Generator, List, Optional, Tuple, Type, Union

from rich.console import Console
from rich.table import Table

class Registry:
    def __init__(self,
                name:str,
                parent: Optional['Registry'] =None,
                scope: Optional[str] =None,
                locations: List=[]
    ):
    self._name = name
    self._module_dict: Dict[str, Type] = dict()
    self._children:Dict[str, 'Registry'] = dict()
    self._locations = locations
    self._imported = False 

    # parent Registry
    self.parent: Optional['Registry']
    if parent is not None:
        assert isinstance(parent, Registry)
        parent._add_child(self)
        self.parent = parent
    else:
        self.parent = None
    
    # build function
    self.build_func: Callable
    if build_func is None:
        if self.parent is not None:
            self.build_func = self.parent.build_func
        else:
            self.build_func = build_from_cfg
    else:
        self.build_func = build_func


    def _add_child(self, registry:Registry) ->None:
        assert(registry, Registry)
        pass
    
    def register_module(self,
                        name: Optional[Union[str, List[str]]] = None,
                        force: bool = False,
                        module: Optional[Type] = None) -> Union[type, Callable]:
    ):
        if not isinstance(force, bool):
            raise TypeError(f'force must be a boolean, but got {type(force)}')
        
        if not (name is None or isinstance(name, str) or is_seq_of(name, str)):
            raise TypeError(
                'name must be None, an instance of str, or a sequence of str, '
                f'but got {type(name)}')

        if module is not None:
            self._register_module(module=module, module_name=name, force=force)
            return module

        def _register(module):
            self._register_module(module=module, module_name=name, force=force)
            return module
        
        return _register

    def register_module(self,
                        module: Type,
                        module_name: Optional[Union[str, List[str]]] = None,
                        force: bool = False
    )->None:
        if not callable(module):
            raise TypeError(f'module must be Callable, but got {type(module)}')

        if module_name is None:
            module_name = module.__name__
        if isinstance(module_name, str):
            module_name = [module_name]
        
        for name in module_name:
            if not force and name in self._module_dict:
                existed_module = self.module_dict[name]
                raise KeyError(f'{name} is already registered in {self.name} '
                               f'at {existed_module.__module__}')
            self._module_dict[name] = module

    def build(self, cfg:dict, *args, *kwargs)-> Any:
        return self.build_func(cfg, *args, **kwargs, registry=self)
        

    def __len__(self):
        return len(self._module_dict)
    
