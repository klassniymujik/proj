import importlib
import sys


def alias_module(wrapper_name: str, target_name: str):
    module = importlib.import_module(target_name)
    sys.modules[wrapper_name] = module
    return module
