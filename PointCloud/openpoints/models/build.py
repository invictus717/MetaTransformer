from openpoints.utils import registry
MODELS = registry.Registry('models')


def build_model_from_cfg(cfg, **kwargs):
    """
    Build a model, defined by `NAME`.
    Args:
        cfg (eDICT): 
    Returns:
        Model: a constructed model specified by NAME.
    """
    return MODELS.build(cfg, **kwargs)