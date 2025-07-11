import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time
from dataset import DatasetQ
from tqdm import tqdm
import argparse
import os
import sys
from MultKAN import KAN

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate curve.')
    parser.add_argument('-seed', type=int, default=1, help='Seed Number')
    parser.add_argument('-k', type=int, default=3, help='Polynomial degree for KAN')
    parser.add_argument('-step', type=int, default=60000, help='Steps to train with')
    parser.add_argument('-lr', type=float, default=0.0001, help='LR')
    parser.add_argument('-data', type=str, default='qd', help='qd, qs, qg')
    parser.add_argument('-tp', type=int, default=1, help='Train part 1,2,4,10')
    parser.add_argument('--h2', action='store_true', help='Test bigger model with 2 hidden layers')
    args = parser.parse_args()

    assert args.data in ['qd', 'qs', 'qg']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    print('Training {} on {} device'.format(args.data, device))

    dataroot = 'data/'
    dataset = DatasetQ(dataroot, args.tp, args.data, device)

    print('Train set length: ', dataset['train_input'].shape[0])
    print('Test set length: ', dataset['test_input'].shape[0])

    checkpoint_dir = './checkpoints_n_k{:d}_{}_2h'.format(args.k, args.data) if args.h2 else './checkpoints_n_k{:d}_{}'.format(args.k, args.data)

    os.makedirs(checkpoint_dir, exist_ok = True)

    grids = np.array([10]) if args.tp == 10 else np.array([16])

    train_losses = []
    test_losses = []

    # Initialize model
    if args.h2:
        model = KAN(width=[2,3,3,1], grid=grids[0], k=args.k, seed=args.seed, device=device, noise_scale=0.1, sp_trainable=False, sb_trainable=False, save_act=True)
    else:
        model = KAN(width=[2,3,1], grid=grids[0], k=args.k, seed=args.seed, device=device, noise_scale=0.1, sp_trainable=False, sb_trainable=False, save_act=True)

    filename = checkpoint_dir + '/model_' + str(args.tp) + '_' + str(args.seed) + '.pt'
    tuned_filename = checkpoint_dir + '/model_' + str(args.tp) + '_' + str(args.seed) + '_tuned.pt'
    try:
        model.load_state_dict(torch.load(filename))
    except:
        print('No model found at: {}'.format(filename))
        sys.exit()
    
    results = model.fit(dataset, opt="Adam", steps=args.step, lr=args.lr, b_coef=0.01)
    train_losses += results['train_loss']
    test_losses += results['test_loss']

    # Save tuned model
    torch.save(model.state_dict(), tuned_filename)

    # Calculate train MAPE
    x = dataset['train_input']
    pred = model(x).detach()
    truth = dataset['train_label']
    mask = torch.abs(truth) > 0.01
    pred = pred[mask]
    truth = truth[mask]
    train_mape = torch.abs((truth - pred) / truth).mean() * 100.0
    print('Train MAPE: ', train_mape.item())

    # Calculate test MAPE
    x = dataset['test_input']
    pred = model(x).detach()
    truth = dataset['test_label']
    mask = torch.abs(truth) > 0.01
    pred = pred[mask]
    truth = truth[mask]
    test_mape = torch.abs((truth - pred) / truth).mean() * 100.0
    print('Test MAPE: ', test_mape.item())

    # Create the results file and write MAPE
    results_file = os.path.join(checkpoint_dir, 'results_{}_{}_tuned.txt'.format(args.tp, args.seed))
    with open(results_file, 'w') as f:
        f.write('Train MAPE: {}'.format(train_mape.item()))
        f.write('Test MAPE: {}'.format(test_mape.item()))