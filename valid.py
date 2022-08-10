import torch
from torchvision.transforms import functional as F
from data import valid_dataloader, test_dataloader
from utils import Adder, calculate_psnr
import os
import numpy as np
from skimage.metrics import peak_signal_noise_ratio
import sys, scipy.io


def _valid(model, config, ep):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataloader = test_dataloader(config.data_dir, batch_size=1, num_workers=0)
    model.eval()
    psnr_adder = Adder()

    with torch.no_grad():
        for idx, data in enumerate(dataloader):
            input_img, label_img = data
            input_img = input_img.to(device)
            if not os.path.exists(os.path.join(config.result_dir, '%d' % (ep))):
                os.mkdir(os.path.join(config.result_dir, '%d' % (ep)))
            pred = model(input_img)

            p_numpy = pred[config.num_subband].squeeze(0).cpu().numpy()
            p_numpy = np.clip(p_numpy, 0, 1)
            in_numpy = label_img.squeeze(0).cpu().numpy()

            psnr = peak_signal_noise_ratio(p_numpy, in_numpy, data_range=1)

            if config.store_opt:
                if ep % config.store_freq == 0:
                    if idx % 20 == 0:
                        save_name = os.path.join(config.result_dir, '%d' %ep, '%d' % (idx) + '.png')
                        save_name_R = os.path.join(config.result_dir, '%d' %ep, '%d' % (idx) + '_Result.png')
                        save_name_I = os.path.join(config.result_dir, '%d' %ep, '%d' % (idx) + '_Input.png')

                        label = F.to_pil_image(label_img.squeeze(0).cpu(), 'RGB')
                        label.save(save_name)

                        input_i = F.to_pil_image(input_img.squeeze(0).cpu(), 'RGB')
                        input_i.save(save_name_I)

                        pred[config.num_subband] = torch.clamp(pred[config.num_subband], 0, 1)
                        result = F.to_pil_image(pred[config.num_subband].squeeze(0).cpu(), 'RGB')
                        result.save(save_name_R)
                        
                        for num_sub in range(config.num_subband):
                            tmp_save_name = os.path.join(config.result_dir, '%d' %ep, '%d' % (idx) + '_' + str(num_sub) + '.mat')
                            tmp_result = pred[num_sub].squeeze(0).cpu().numpy()
                            scipy.io.savemat(tmp_save_name, mdict={'data': tmp_result})

            psnr_adder(psnr)
            if idx % 100 == 0:
                print('\r%03d'%idx, end=' ')
    print('\n')
    model.train()
    return psnr_adder.average()

