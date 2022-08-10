import os
import torch
from torchvision.transforms import functional as F
import numpy as np
from utils import Adder, calculate_psnr
from data import test_dataloader
from utils import EvalTimer
from skimage.metrics import peak_signal_noise_ratio
import time
import sys, scipy.io

def _eval(model, config):
    model_pretrained = os.path.join('results/', config.model_name, config.test_model)
    state_dict = torch.load(model_pretrained)
    model.load_state_dict(state_dict['model'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataloader = test_dataloader(config.data_dir, batch_size=1, num_workers=0)
    adder = Adder()
    model.eval()
    with torch.no_grad():
        psnr_adder = Adder()
        for iter_idx, data in enumerate(dataloader):
            input_img, label_img = data

            input_img, label_img = data
            input_img = input_img.to(device)
            
            tm = time.time()
            pred = model(input_img)
            elaps = time.time() - tm
            adder(elaps)

            p_numpy = pred[config.num_subband].squeeze(0).cpu().numpy()
            p_numpy = np.clip(p_numpy, 0, 1)
            in_numpy = label_img.squeeze(0).cpu().numpy()

            psnr = peak_signal_noise_ratio(p_numpy, in_numpy, data_range=1)

            save_name = os.path.join(config.result_dir, '%d' % (iter_idx) + '.png')
            save_name_R = os.path.join(config.result_dir, '%d' % (iter_idx) + '_Result.png')
            save_name_I = os.path.join(config.result_dir, '%d' % (iter_idx) + '_Input.png')

            label = F.to_pil_image(label_img.squeeze(0).cpu(), 'RGB')
            label.save(save_name)

            input_i = F.to_pil_image(input_img.squeeze(0).cpu(), 'RGB')
            input_i.save(save_name_I)

            pred[config.num_subband] = torch.clamp(pred[config.num_subband], 0, 1)
            result = F.to_pil_image(pred[config.num_subband].squeeze(0).cpu(), 'RGB')
            result.save(save_name_R)
            
            for num_sub in range(config.num_subband):
                tmp_save_name = os.path.join(config.result_dir, '%d' % (iter_idx) + '_' + str(num_sub) + '.mat')
                tmp_result = pred[num_sub].squeeze(0).cpu().numpy()
                scipy.io.savemat(tmp_save_name, mdict={'data': tmp_result})
                
                tmp_save_name_png = os.path.join(config.result_dir, '%d' % (iter_idx) + '_' + str(num_sub) + '.png')
                tmp_result_png = torch.clamp(pred[num_sub], -1, 1)
                tmp_result_png = (tmp_result_png.squeeze(0).cpu() + 1) / 2;
                tmp_result_png = F.to_pil_image(tmp_result_png, 'RGB')
                tmp_result_png.save(tmp_save_name_png)
                
            psnr = peak_signal_noise_ratio(p_numpy, in_numpy, data_range=1)
            psnr_adder(psnr)
            print('%d iter PSNR: %.2f time: %f' % (iter_idx + 1, psnr, elaps))

        print('The average PSNR is %.2f dB' % (psnr_adder.average()))
        print("Average time: %f"%adder.average())
