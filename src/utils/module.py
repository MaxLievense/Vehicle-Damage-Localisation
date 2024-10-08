from importlib import import_module


def import_from_cfg(cfg: str) -> callable:
    """
    Allows to import a class from the Hydra format "module.class"
    """
    module_name, class_name = cfg.rsplit(".", 1)
    module = import_module(module_name)
    return getattr(module, class_name)
