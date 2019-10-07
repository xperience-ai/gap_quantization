import copy

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url


def load_partial_weights(model, checkpoint):
    new_ckpt_state_dict = {}
    for key in checkpoint.keys():
        if key in model.state_dict().keys():
            if checkpoint[key].shape == model.state_dict()[key].shape:
                new_ckpt_state_dict[key] = copy.deepcopy(checkpoint[key])
                print("Successfully loaded all parameters for", key)
            else:
                new_ckpt_state_dict[key] = copy.deepcopy(model.state_dict()[key])
                in_shapes = checkpoint[key].shape
                out_shapes = model.state_dict()[key].shape
                assert len(in_shapes) == len(out_shapes)

                min_shape = [min(in_shape, out_shape) for in_shape, out_shape in zip(in_shapes, out_shapes)]
                if len(min_shape) in [2, 4]:  # Conv2d or FC
                    new_ckpt_state_dict[key][:min_shape[0], :min_shape[1], ...] = \
                        copy.deepcopy(checkpoint[key])[:min_shape[0], :min_shape[1], ...]
                elif len(min_shape) == 1:  # Other
                    new_ckpt_state_dict[key][:min_shape[0]] = copy.deepcopy(checkpoint[key])[:min_shape[0]]
                print("Successfully loaded part of the parameters for", key)

    for key in model.state_dict().keys():
        if key not in new_ckpt_state_dict.keys():
            print('Could not find parameters named {} in checkpoint'.format(key))
            new_ckpt_state_dict[key] = copy.deepcopy(model.state_dict()[key])

    model.load_state_dict(new_ckpt_state_dict)
    print('Successfully finished partial loading of model')
    return model
