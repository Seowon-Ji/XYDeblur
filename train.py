import os
import torch

from data import train_dataloader
from utils import Adder, Timer, check_lr
from torch.utils.tensorboard import SummaryWriter
from valid import _valid
import torch.nn.functional as F
from tqdm import tqdm

def _train(model, config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=config.learning_rate,
                                 weight_decay=config.weight_decay)

    dataloader = train_dataloader(config.data_dir, config.batch_size, config.num_worker)
    max_iter = len(dataloader)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, config.lr_steps, config.gamma)

    writer = SummaryWriter(os.path.join('runs',config.model_name))
    epoch_adder = Adder()
    iter_adder = Adder()
    epoch_timer = Timer('m')
    iter_timer = Timer('m')

    model_save_overwrite = os.path.join(config.model_save_dir, 'model_overwrite.pkl')

    if os.path.isfile(model_save_overwrite):
        state_dict = torch.load(model_save_overwrite)
        model.load_state_dict(state_dict['model'])
        optimizer.load_state_dict(state_dict["optimizer"])
        scheduler.load_state_dict(state_dict["scheduler"])
        start_epoch = state_dict['epoch']
        lr = check_lr(optimizer)
        print("\n Model Restored, epoch = %4d\n" % (start_epoch + 1))
        print("\n                 current lr = %10f\n" % (lr))
    else:
        print("No previous data... Started from scratch ... \n")
        start_epoch = 0

    best_psnr=-1
    for epoch_idx in range(start_epoch + 1, config.num_epoch + 1):
        epoch_timer.tic()
        iter_timer.tic()
        for iter_idx, batch_data in enumerate(tqdm(dataloader)):

            input_img, label_img = batch_data
            input_img = input_img.to(device)
            label_img = label_img.to(device)

            optimizer.zero_grad()
            pred_img = model(input_img)

            l_tot = criterion(pred_img[config.num_subband], label_img)

            loss = l_tot
            
            loss.backward()
            optimizer.step()

            iter_adder(loss.item())
            epoch_adder(loss.item())

            if (iter_idx + 1) % config.print_freq == 0:
                lr = check_lr(optimizer)
                print("Time: %7.4f Epoch: %03d Iter: %4d/%4d LR: %.10f Loss: %7.4f" % (iter_timer.toc(), epoch_idx,
                                                                             iter_idx + 1, max_iter, lr,
                                                                             iter_adder.average()))
                writer.add_scalar('Loss', iter_adder.average(), iter_idx + (epoch_idx-1)* max_iter)
                iter_timer.tic()
                iter_adder.reset()

        if epoch_idx % config.save_freq == 0:
            torch.save({'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'epoch': epoch_idx}, model_save_overwrite)

        print("EPOCH: %02d\nElapsed time: %4.2f Epoch Loss: %7.4f" % (
        epoch_idx, epoch_timer.toc(), epoch_adder.average()))
        epoch_adder.reset()
        scheduler.step()
        if epoch_idx % config.valid_freq == 0:
            val = _valid(model, config, epoch_idx)
            print('%03d epoch \n Average PSNR %.2f dB' % (epoch_idx, val))
            writer.add_scalar('PSNR', val, epoch_idx)

            if val >= best_psnr:
                torch.save({'model': model.state_dict()}, os.path.join(config.model_save_dir, 'Best.pkl'))

            save_name = os.path.join(config.model_save_dir, 'model_%d.pkl' % epoch_idx)
            torch.save({'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'epoch': epoch_idx}, save_name)
    
    save_name = os.path.join(config.model_save_dir, 'Final.pkl')
    torch.save({'model': model.state_dict()}, save_name)
