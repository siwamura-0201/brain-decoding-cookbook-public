# DNN feature extraction from images

Extraction of DNN features from stimulus images. The features produced here are
the input of the feature decoding scripts in
[feature-decoding](../feature-decoding) (`decoder.features.paths`) and of the
reconstruction scripts when true features are used instead of decoded ones.

```
images --> [feature-extraction] --> features --> [feature-decoding] --> decoded features --> [reconstruction]
```

## Setup

### Downloading the encoder

Run the following in the `data` directory. This is the same archive used by the
reconstruction analysis, so it can be skipped if it has already been downloaded.

``` shellsession
$ uv run download.py recon_demo
```

This puts the encoder parameters and the ImageNet mean image in
`data/models/VGG_ILSVRC_19_layers`.

### Preparing the images

**The stimulus images are not distributed with this repository** (see the
license of the source image datasets). They need to be placed by hand, in the
location given by `images.path` in the configuration file:

```
data/images/ImageNetTest/source/*.JPEG
```

The file name without the extension is used as the feature label, and it has to
match the stimulus label in the fMRI data (`stimulus_name`) for the features to
be usable by the decoding scripts.

## Usage

``` shellsession
$ uv run extract_features.py config/extract_features_vgg19_ImageNetTest.yaml
```

Features are written as `<features.path>/<layer>/<image label>.mat` with the
array key `feat`, which is the layout read by `bdpy.dataform.Features`. Images
whose features are already saved for every layer are skipped, so the script can
be interrupted and resumed.

To extract on the CPU, or to forward several images at once:

``` shellsession
$ uv run extract_features.py config/extract_features_vgg19_ImageNetTest.yaml --override device=cpu batch_size=8
```

## Caffe features and PyTorch features

The features distributed via figshare (`features-ImageNet*-caffe-VGG19-*.zip`,
downloaded by `feature-decoding/data/download.py`) were extracted with the
original Caffe implementation, and the decoders distributed with them were
trained on those features. This script uses the PyTorch port of the same
network (`VGG_ILSVRC_19_layers.pt`, converted from
`VGG_ILSVRC_19_layers.caffemodel`), which is **not guaranteed to reproduce the
Caffe features exactly**.

The output is therefore written under `pytorch/VGG19` rather than
`caffe/VGG19`, following the `<framework>/<network>` convention of the feature
store. Before mixing these features with the distributed ones, check how far
apart they are:

``` shellsession
$ uv run compare_features.py \
    ./data/features/ImageNetTest/caffe/VGG19 \
    ./data/features/ImageNetTest/pytorch/VGG19
```

`compare_features.py` reports, for each layer, the correlation, the relative
RMSE, and the largest absolute difference between the two feature sets. If the
agreement is close enough for the analysis at hand, set `features.name` to
`caffe/VGG19` in the configuration file.

## Preprocessing

The preprocessing follows the original Caffe pipeline rather than the
torchvision convention:

- The image is resized to the input size of the encoder (224 x 224) with bicubic
  interpolation. The aspect ratio is **not** preserved and no cropping is done.
- Pixel values are kept in the 0-255 range; they are **not** scaled to [0, 1].
- Channels are ordered BGR.
- The channel-wise mean of the ImageNet training images (`image_mean_file`,
  reduced to one value per channel) is subtracted.

Replacing any of these by the torchvision defaults (`ToTensor`, RGB,
`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`) produces features
that are not comparable with the ones used in this cookbook.

## Adding another encoder

`encoder.name` is passed to `bdpy.dl.torch.models.model_factory` and
`layer_map`, so any network supported by bdpy can be used by adding a
configuration file to `config/encoder` with `name`, `parameters_file`,
`input_image_shape`, `image_mean_file` (or `image_mean`, a list of three
channel means in BGR order), and `layers`. Layer names are the human-readable
ones of `layer_map` (`conv1_1`, `fc6`, ...), which are also the directory names
in the feature store.

## Note

`data` and `models` are symbolic links to `../data` and `../data/models`. On a
platform without symbolic link support they are checked out as plain text
files, and the paths in the configuration files have to be adjusted.
