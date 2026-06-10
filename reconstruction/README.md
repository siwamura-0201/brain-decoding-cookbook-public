# Reconstruction with decoded DNN features

## Setup

### Downloading data

Run the following in `data` directory.

``` shellsession
$ uv run download.py recon_demo
```

## Usage

### iCNN reconstruction

Run the following command.

``` shellsession
$ python recon_icnn_image_gd.py config/recon_icnn_vgg19_relu7generator_gd_1000iter_decoded_ImageNet.yaml
```

This will output reconstructed images in `./data/reconstruction/icnn/recon_icnn_image_gd_vgg19_relu7generator_scaling_feature_std_train_mean_center_1000iter/decodedImageNetTest_deeprecon_VGG19`.

If you want to change the reconstruction parameters at run time, please use `--override` option.

``` shellsession
# Use Shen scaling

$ python recon_icnn_image_gd.py config/recon_icnn_vgg19_relu7generator_gd_1000iter_decoded_ImageNet.yaml  --override icnn.feature_scaling=feature_std_shen_original

# Use raw decoded features ('null' is convert to None in Python script.)

$ python recon_icnn_image_gd.py config/recon_icnn_vgg19_relu7generator_gd_1000iter_decoded_ImageNet.yaml  --override icnn.feature_scaling=null
```

### Evaluation

When evaluating the reconstructed images, use the `--analysis` option and specify the name of the reconstruction script. 

``` shellsession
$ python recon_eval_image.py config/recon_icnn_vgg19_relu7generator_gd_1000iter_decoded_ImageNet.yaml --analysis recon_icnn_image_gd_dist
```


## Issues

We have noticed that the code is not functioning properly with the following versions of PyTorch. Currently, we are working on debugging the issue.

- PyTorch 1.9.1
- PyTorch 1.9.0

## Appendix

- Data files are hosted at <https://figshare.com/articles/dataset/brain-decoding-cookbook/21564384>.
- The code was tested in the following environments.
  - Python 3.10 + PyTorch 1.13.1 + CUDA 11.6
  - Python 3.10 + PyTorch 1.12.1 + CUDA 11.6
  - Python 3.8 + PyTorch 1.7.1 + CUDA 10.1
  - Docker: [pytorch/pytorch:1.7.1-cuda11.0-cudnn8-runtime](https://hub.docker.com/layers/pytorch/pytorch/1.7.0-cuda11.0-cudnn8-runtime/images/sha256-9cffbe6c391a0dbfa2a305be24b9707f87595e832b444c2bde52f0ea183192f1)