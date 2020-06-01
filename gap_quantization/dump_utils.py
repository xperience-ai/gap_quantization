import json
import os
import os.path as osp
import re
import shutil

import torch
import torch.nn as nn

#from urllib.parse import urlparse


def create_norm_list(save_dir, convbn):
    dict_norm = {}
    for file in os.listdir(save_dir):
        if file != 'activations_dump' and not re.match('.*cat.json', file) and not re.match('.*txt', file):
            dict_norm[file] = read_norm(os.path.join(save_dir, file))
    list_norm = make_list_from_dict(dict_norm, convbn)
    txt_list = open('param_layer_quant.h', 'w')
    for i in range(0, 26):
        txt_list.write("#define Q_IN_" + '{:<8}'.format(str(i)) + str(list_norm[i]['q_in']) + "\n")
        txt_list.write("#define Q_OUT_" + '{:<7}'.format(str(i)) + str(list_norm[i]['q_out']) + "\n")
        txt_list.write("#define Q_WEIGHTS_" + '{:<3}'.format(str(i)) + str(list_norm[i]['q_weights']) + "\n")
        txt_list.write("#define Q_BIAS_" + '{:<6}'.format(str(i)) + str(list_norm[i]['q_bias']) + "\n")
        txt_list.write("\n")


def read_norm(file):
    print(file)
    with open(file, "rt") as js_file:
        data = json.load(js_file)
    param_dict = {}
    param_dict['norm'] = int(data['norm'][0])
    param_dict['q_in'] = int(data['inp_frac_bits'][0])
    param_dict['q_out'] = int(data['out_frac_bits'][0])
    param_dict['q_weights'] = int(data['w_frac_bits'][0])
    param_dict['q_bias'] = int(data['b_frac_bits'][0])
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
