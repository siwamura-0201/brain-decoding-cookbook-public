# DNN feature extraction from images

Extraction of DNN features from stimulus images. The features are used as true features in [feature-decoding](../feature-decoding) and [reconstruction](../reconstruction).

## Setup

### Downloading data

Run the following in `data` directory. This is the same archive as the one used by the reconstruction analysis, so it can be skipped if you have already downloaded it.

``` shellsession
$ uv run download.py recon_demo
```

This will put the encoder parameters and the ImageNet mean image in `./data/models/VGG_ILSVRC_19_layers`.

### Preparing stimulus images

The stimulus images are not distributed with this repository. Place them by hand in the directory specified by `images.path` in the config file.

```
data/images/ImageNetTest/source/*.JPEG
```

The file name without its extension is used as the feature label, and it has to match the stimulus name in the fMRI data (`stimulus_name`).

## Usage

Run the following command.

``` shellsession
$ uv run extract_features.py config/extract_features_vgg19_ImageNetTest.yaml
```

This will output features in `./data/features/ImageNetTest/pytorch/VGG19/<layer>/<image label>.mat`, the layout read by `bdpy.dataform.Features`. Images whose features are already saved for every layer are skipped, so the script can be resumed after an interruption.

If you want to change the extraction parameters at run time, please use `--override` option.

``` shellsession
# Extract on the CPU, forwarding 8 images at once

$ uv run extract_features.py config/extract_features_vgg19_ImageNetTest.yaml --override device=cpu batch_size=8
```

## Appendix

- The distributed features were extracted with the original Caffe implementation, while this script uses its PyTorch port, and the two are not guaranteed to be identical. The output is therefore stored under `pytorch/VGG19`, separately from the distributed `caffe/VGG19`.
- The preprocessing follows the original Caffe pipeline (224 x 224 bicubic resize without preserving the aspect ratio, pixel values kept in the 0-255 range, BGR channel order, and subtraction of the channel-wise ImageNet mean), not the torchvision convention. Replacing it with the torchvision defaults produces features that are not comparable with the ones used in this cookbook.
- Another encoder can be used by adding a config file to `config/encoder`; `encoder.name` is passed to `bdpy.dl.torch.models.model_factory`, and the layer names are the ones of its `layer_map` (`conv1_1`, `fc6`, ...).
- `data` and `models` are symbolic links to `../data` and `../data/models`.
