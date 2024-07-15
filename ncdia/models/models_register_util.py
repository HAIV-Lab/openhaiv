from typing import Any, Callable, List, Optional, TypeVar
from torch import nn
from types import ModuleType

# 定义一个注册model方法
BUILTIN_MODELS=[]
H = TypeVar("H", bound=nn.Module)
def register_model(name: Optional[str] = None) -> Callable[[Callable[..., H]], Callable[..., H]]:
    def wrapper(fn: Callable[..., H]) -> Callable[..., H]:
        key = name if name is not None else fn.__name__
        if key in BUILTIN_MODELS:
            raise ValueError(f"An model is already registered under the name '{key}'.")
        BUILTIN_MODELS[key] = fn
        return fn
    return wrapper

# 定义一个获取已注册model列表的方法
def list_models(module: Optional[ModuleType] = None) -> List[str]:
    """
    Returns a list with the names of registered models.

    .. betastatus:: function

    Args:
        module (ModuleType, optional): The module from which we want to extract the available models.

    Returns:
        models (list): A list with the names of available models.
    """
    models = [
        k for k, v in BUILTIN_MODELS.items() if module is None or v.__module__.rsplit(".", 1)[0] == module.__name__
    ]
    return sorted(models)
# 定义一个根据名称获取model的方法
def get_model_builder(name: str) -> Callable[..., nn.Module]:
    """
    Gets the model name and returns the model builder method.

    .. betastatus:: function

    Args:
        name (str): The name under which the model is registered.

    Returns:
        fn (Callable): The model builder method.
    """
    name = name.lower()
    try:
        fn = BUILTIN_MODELS[name]
    except KeyError:
        raise ValueError(f"Unknown model {name}")
    return fn

# 定义一个根据名称和配置获取model的方法
def get_model(name: str, **config: Any) -> nn.Module:
    """
    Gets the model name and configuration and returns an instantiated model.

    .. betastatus:: function

    Args:
        name (str): The name under which the model is registered.
        **config (Any): parameters passed to the model builder method.

    Returns:
        model (nn.Module): The initialized model.
    """
    fn = get_model_builder(name)
    return fn(**config)
