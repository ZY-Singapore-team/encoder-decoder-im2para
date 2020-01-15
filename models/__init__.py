from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import copy

import numpy as np
import misc.utils as utils
import torch

from .HierModel import HTopDownModel
from .AttModel import TopDownModel

def setup(opt):
    if opt.caption_model == 'hierarchical':
    	print('hierarchical model training...')
    	model = HTopDownModel(opt)
    else:
    	print('SCT with BERT model training...')
    	model = TopDownModel(opt)

    # Check compatibility if training is continued from previously saved model
    if vars(opt).get('start_from', None) is not None:
        # check if all necessary files exist 
        assert os.path.isdir(opt.checkpoint_path)," %s must be a a path" % opt.start_from
        assert os.path.isfile(os.path.join(opt.checkpoint_path, 'infos_'
                + opt.id + format(int(opt.start_from),'04') + '.pkl')),"infos.pkl file does not exist in path %s"%opt.start_from
        model.load_state_dict(torch.load(os.path.join(
            opt.checkpoint_path, 'model' +opt.id+ format(int(opt.start_from),'04') + '.pth')))

    return model