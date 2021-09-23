import os
import sys
import time
from datetime import datetime

from IO import config as usr_config
from IO import dconst
from nn import config as nn_config


def blockPrinting(func):
    def func_wrapper(*args, **kwargs):
        # block all printing to the console
        sys.stdout = open(os.devnull, 'w')
        # call the method in question
        value = func(*args, **kwargs)
        # enable all printing to the console
        sys.stdout = sys.__stdout__
        # pass the return value of the method back
        return value

    return func_wrapper


@blockPrinting  # comment this out if you want to see the print out in each trial
def run(usr_configs):
    trainer_configs = nn_config.TrainerConfigs()
    trainer_configs.setup(usr_configs)
    trainer = nn_config.get_trainer(usr_configs.train)(trainer_configs)
    trainer.train()
    return trainer.model_dst_path, trainer.nnresult.test_accuracy


if __name__ == '__main__':
    fmsize=[24, 32, 48, 56, 64]

    for f in fmsize:
        usr_configs = usr_config.get_usr_config()
        usr_configs.model.init_args.fmsize = f
        usr_configs.train.result_dir = os.path.join('/data/shanbhag/hgeng4/MSTAR/atrlite_fl', str(f))
        parent_result_dir = usr_configs.train.result_dir

        for i in range(dconst.NUM_TIME_TRAIN):
            usr_configs.seed = dconst.SEED_TRAIN[i]
            usr_configs.train.result_dir = os.path.join(parent_result_dir, str(i))

            if i > 0:
                usr_config.profile_complexity = False

            start = datetime.now()
            print('starting trial {} at {}'.format(i, start))
            model_dst_path, final_acc = run(usr_configs)
            end = datetime.now()
            print('trial {} finished in {}h'.format(i, (end - start).seconds/3600))
            print('model saved at {}'.format(model_dst_path))
            print('fional model acc: {}'.format(final_acc))
