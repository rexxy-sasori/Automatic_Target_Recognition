import argparse

import torch

from nn import config as nn_config


def main():
    parser = argparse.ArgumentParser(description='model evaluation script')
    parser.add_argument('model_src_path', type=str, help='model src path')
    args = parser.parse_args()
    ckpt = torch.load(args.model_src_path)

    usr_configs = ckpt.get('analysis').trainer_usr_configs
    usr_configs.train.train_model = False
    usr_configs.train.model_src_path = args.model_src_path
    trainer_configs = nn_config.TrainerConfigs()
    trainer_configs.setup(usr_configs)

    trainer = nn_config.get_trainer(usr_configs.train)(trainer_configs)
    trainer.eval(trainer.trainer_configs.test_loader, print_acc=True, cfm=True)


if __name__ == '__main__':
    main()
