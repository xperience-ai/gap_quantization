import json
import os
import os.path as osp
import re
import shutil

import torch
import torch.nn as nn

#from urllib.parse import urlparse


def dump_quant_params(save_dir, convbn):
    dict_norm = {}
    for file in os.listdir(save_dir):
        if file != 'activations_dump' and not re.match('.*cat.json', file) and not re.match('.*txt', file):
            dict_norm[file] = read_quant_params(os.path.join(save_dir, file))
    list_norm = make_list_from_dict(dict_norm, convbn)
    with open('param_layer_quant.h', 'w') as txt_list:
        for i in range(0, 26):
            print("#define Q_IN_{:<8}{}".format(i, list_norm[i]['inp_frac_bits']), file=txt_list)
            print("#define Q_OUT_{:<7}{}".format(i, list_norm[i]['out_frac_bits']), file=txt_list)
            print("#define Q_WEIGHTS_{:<3}{}".format(i, list_norm[i]['w_frac_bits']), file=txt_list)
            print("#define Q_BIAS_{:<6}{}".format(i, list_norm[i]['b_frac_bits']), file=txt_list)
            print("", file=txt_list)


def read_quant_params(file):
    print(file)
    with open(file) as json_file:
        data = json.load(json_file)
    param_dict = {key: int(val[0]) for (key, val) in data.items() if key not in ('weight' or 'bias')}
    return param_dict


def make_list_from_dict(param_dict, bn):
    param_list = []
    if bn:
        param_list.extend([param_dict['conv1.0.json'], param_dict['features.0.0.json']])
        features_names_list = ['.squeeze.0.json', '.expand1x1.0.json', '.expand3x3.0.json']
    else:
        param_list.extend([param_dict['conv1.json'], param_dict['features.0.json']])
        features_names_list = ['.squeeze.json', '.expand1x1.json', '.expand3x3.json']
    for i in [3, 4, 6, 7, 9, 10, 11, 12]:
        for feature_name in features_names_list:
            param_list.extend([param_dict['features.' + str(i) + feature_name]])
    return param_list


def remove_extra_dump(directory):
    extra_list = ['conv1', 'conv1.1', 'features.0', 'features.0.1']
    features_names_list = [
        '.squeeze', '.squeeze.1', '.expand1x1', '.expand1x1.1', '.expand3x3', '.expand3x3.1'
    ]
    for i in [3, 4, 6, 7, 9, 10, 11, 12]:
        for feature_name in features_names_list:
            extra_list.append('features.' + str(i) + feature_name)
    for folder in os.listdir(directory):
        if folder in extra_list:
            shutil.rmtree(os.path.join(directory, folder))


def remove_cat_files(directory):
    for i in [3, 4, 6, 7, 9, 10, 11, 12]:
        os.remove(os.path.join(directory, 'features.' + str(i) + ".cat.json"))
