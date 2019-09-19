import copy
import re

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url
def get_model_name(checkpoint_keys, model_keys):
    mapping = {}
    checkpoint_keys = sorted(checkpoint_keys)
    model_keys = sorted(model_keys)
    j = 0
    for i in range(len(checkpoint_keys)):
        if re.sub(r'[0-9]+', '', checkpoint_keys[i]) == re.sub(r'[0-9]+', '', model_keys[j]):
        #if checkpoint_keys[i] == model_keys[j]:
            mapping[model_keys[j]] = checkpoint_keys[i]
        else:
            i -= 1
        j += 1
    return  mapping

def load_weights(model, checkpoint):
    new_ckpt_state_dict = {}
    mapping = get_model_name(checkpoint.keys(), model.state_dict().keys())
    for key in mapping.keys():
            if checkpoint[mapping[key]].shape == model.state_dict()[key].shape:
                new_ckpt_state_dict[key] = copy.deepcopy(checkpoint[mapping[key]])
                print("Successfully loaded all parameters for", key)
            else:
                new_ckpt_state_dict[key] = copy.deepcopy(model.state_dict()[key])
                in_shape = checkpoint[mapping[key]].shape
                out_shape = model.state_dict()[key].shape
                assert len(in_shape) == len(out_shape)

                min_shape = [min(in_shape[i], out_shape[i]) for i in range(len(in_shape))]
                try:
                    if len(min_shape) == 4:  # Conv2d
                        new_ckpt_state_dict[key][:min_shape[0], :min_shape[1], :, :] = \
                            copy.deepcopy(checkpoint[mapping[key]])[:min_shape[0], :min_shape[1], :, :]
                    if len(min_shape) == 1:  # Other
                        new_ckpt_state_dict[key][:min_shape[0]] = copy.deepcopy(checkpoint[mapping[key]])[:min_shape[0]]
                    print("Successfully loaded part of the parameters for", key)
                except:
                    print("Unsuccessful trial to partially load parameters for", key)

    for key in model.state_dict().keys():
        if key not in new_ckpt_state_dict.keys():
            print('Could not find parameters named {} in checkpoint'.format(key))
            new_ckpt_state_dict[key] = copy.deepcopy(model.state_dict()[key])

    model.load_state_dict(new_ckpt_state_dict)
    print('Successfully finished partial loading of model')
    return model