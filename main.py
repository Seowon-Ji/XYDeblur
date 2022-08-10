import os
import torch
import argparse
from torch.backends import cudnn
from models.XYDeblur import build_net
from train import _train
from eval import _eval


def main(config):
    cudnn.benchmark = True

    if not os.path.exists('results/'):
        os.makedirs(config.model_save_dir)
    if not os.path.exists('results/' + config.model_name + '/'):
        os.makedirs('results/' + config.model_name + '/')
    if not os.path.exists(config.model_save_dir):
        os.makedirs(config.model_save_dir)
    if not os.path.exists(config.result_dir):
        os.makedirs(config.result_dir)

    model = build_net()
    if torch.cuda.is_available():
        model.cuda()
    if config.mode == 'train':
        _train(model, config)

    elif config.mode == 'test':
        _eval(model, config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Directories
    parser.add_argument('--model_name', type=str, default='XYDeblur')
    parser.add_argument('--data_dir', type=str, default='./sample_data')
    
    # Train
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--num_epoch', type=int, default=3000)
    parser.add_argument('--print_freq', type=int, default=100)
    parser.add_argument('--num_worker', type=int, default=8)
    parser.add_argument('--save_freq', type=int, default=50)
    parser.add_argument('--valid_freq', type=int, default=100)
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--lr_steps', type=list, default=[(x+1) * 500 for x in range(10000//500)])

    parser.add_argument('--store_opt', type=bool, default=True)
    parser.add_argument('--num_subband', type=int, default=2)
    parser.add_argument('--store_freq', type=int, default=300)

    # Test
    parser.add_argument('--test_model', type=str, default='pretrained_model.pkl')
    parser.add_argument('--mode', type=str, default='test')

    config = parser.parse_args()
    config.model_save_dir = os.path.join('results/', config.model_name, 'weights/')
    config.result_dir = os.path.join('results/', config.model_name, 'eval/')
    print(config)
    main(config)
