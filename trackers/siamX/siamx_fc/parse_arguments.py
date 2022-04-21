import json
from collections import namedtuple
import sys
sys.path.append('../')

try:
    from _paramparse_ import paramparse
except ModuleNotFoundError:
    import paramparse

def parse_arguments(in_hp={}, in_evaluation={}, in_run={}):

    with open('trackers/siamx/siamx_fc/parameters/hyperparams.json') as json_file:
        hp = json.load(json_file)
    # with open('./siamx_fc/parameters/evaluation.json') as json_file:
    #     evaluation = json.load(json_file)
    # with open('./siamx_fc/parameters/run.json') as json_file:
    #     run = json.load(json_file)
    # with open('./siamx_fc/parameters/environment.json') as json_file:
    #     env = json.load(json_file)
    with open('trackers/siamx/siamx_fc/parameters/design.json') as json_file:
        design = json.load(json_file)

    paramparse.from_dict(hp, 'siamx_fc_hp')
    # paramparse.from_dict(evaluation, 'siamx_fc_evaluation')
    # paramparse.from_dict(run, 'siamx_fc_run')
    # paramparse.from_dict(env, 'siamx_fc_env')
    paramparse.from_dict(design, 'siamx_fc_design')

    # for name,value in in_hp.items():
    #     hp[name] = value
    # for name,value in in_evaluation.items():
    #     evaluation[name] = value
    # for name,value in in_run.items():
    #     run[name] = value
    
    hp = namedtuple('hp', hp.keys())(**hp)
    # evaluation = namedtuple('evaluation', evaluation.keys())(**evaluation)
    # run = namedtuple('run', run.keys())(**run)
    # env = namedtuple('env', env.keys())(**env)
    design = namedtuple('design', design.keys())(**design)

    return hp, design
