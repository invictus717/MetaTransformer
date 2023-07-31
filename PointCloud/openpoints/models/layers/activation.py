from torch import nn
import copy


_ACT_LAYER = dict(
    silu=nn.SiLU,
    swish=nn.SiLU,
    mish=nn.Mish,
    relu=nn.ReLU,
    relu6=nn.ReLU6,
    leaky_relu=nn.LeakyReLU,
    leakyrelu=nn.LeakyReLU,
    elu=nn.ELU,
    prelu=nn.PReLU,
    celu=nn.CELU,
    selu=nn.SELU,
    gelu=nn.GELU,
    sigmoid=nn.Sigmoid,
    tanh=nn.Tanh,
    hard_sigmoid=nn.Hardsigmoid,
    hard_swish=nn.Hardswish,
)


def create_act(act_args):
    """Build activation layer.
    Returns:
        nn.Module: Created activation layer.
    """
    if act_args is None:
        return None
    act_args = copy.deepcopy(act_args)
    
    if isinstance(act_args , str):
        act_args = {"act": act_args}    
    
    act = act_args.pop('act', None)
    if act is None:
        return None

    if isinstance(act, str):
        act = act.lower()
        assert act in _ACT_LAYER.keys(), f"input {act} is not supported"
        act_layer = _ACT_LAYER[act]

    inplace = act_args.pop('inplace', True)

    if act not in ['gelu', 'sigmoid']: # TODO: add others
        return act_layer(inplace=inplace, **act_args)
    else:
        return act_layer(**act_args)


if __name__ == "__main__":
    act_args = {'act': 'relu', 'inplace': False}
    act_layer = create_act(act_args)
    print(act_layer)
