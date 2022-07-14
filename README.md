# XYDeblur: Offcial Pytorch Implementation

This repository provides the official PyTorch implementation of the following paper:

> XYDeblur: Divide and Conquer for Single Image Deblurring
> In CVPR 2022. (* indicates equal contribution)
>
> Paper: [XYDeblur]([http://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html](https://openaccess.thecvf.com/content/CVPR2022/html/Ji_XYDeblur_Divide_and_Conquer_for_Single_Image_Deblurring_CVPR_2022_paper.html) "XYDeblur")
>
> Abstract: Many convolutional neural networks (CNNs) for single image deblurring employ a U-Net structure to estimate latent sharp images. Having long been proven to be effective in image restoration tasks, a single lane of encoder-decoder architecture overlooks the characteristic of deblurring, where a blurry image is generated from complicated blur kernels caused by tangled motions. Toward an effective network architecture, we present complemental sub-solutions learning with a one-encoder-two-decoder architecture for single image deblurring. Observing that multiple decoders successfully learn to decompose information in the encoded features into directional components, we further improve both the network efficiency and the deblurring performance by rotating and sharing kernels exploited in the decoders, which prevents the decoders from separating unnecessary components such as color shift. As a result, our proposed network shows superior results as compared to U-Net while preserving the network parameters, and the use of the proposed network as the base network can improve the performance of existing state-of-the-art deblurring networks.

---
