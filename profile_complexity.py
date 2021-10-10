import argparse

import torch

from nn import config as nn_config

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="script for profiling complexity")
    parser.add_argument("model_src_path")
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ckpt = torch.load(args.model_src_path, map_location=device)

    usr_configs = ckpt.get('analysis').trainer_usr_configs
    usr_configs.train.train_model = False
    usr_configs.device.use_gpu = False if device == 'cpu' else True

    usr_configs.profile_complexity = True

    usr_configs.train.model_src_path = args.model_src_path
    trainer_configs = nn_config.TrainerConfigs()
    trainer_configs.setup(usr_configs)

    trainer = nn_config.get_trainer(usr_configs.train)(trainer_configs)
    ckpt['analysis'] = trainer.nnresult
    torch.save(ckpt, args.model_src_path)
    print('Done writing profiling result to tar file')