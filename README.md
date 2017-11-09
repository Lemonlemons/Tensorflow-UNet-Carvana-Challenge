# Carvana Tensorflow UNet Implementation

This implementation was based on [Heng CherKeng's code for PyTorch](https://www.kaggle.com/c/carvana-image-masking-challenge/discussion/37208)
and on [Peter Giannakopoulos's code for Keras](https://github.com/petrosgk/Kaggle-Carvana-Image-Masking-Challenge). Big props to both of them.

## Requirements
* Tensorflow
* sklearn
* cv2

## Usage
### Convert masks
Place '*train*', '*train_masks*' and '*test*' data folders in the '*input*' folder.

Convert training masks to *.png* format. 

You can do this by downloading ImageMagick (which includes mogrify) from this location:

https://www.imagemagick.org/script/download.php

then running this command in the train_masks directory

` magick mogrify -format png *.gif` 

### Preprocessing and creating .tfrecords for easy consumption
Run `python main.py --phase preprocess --model unet1024`

### Train
Run `python main.py --phase train --model unet1024`

### Test
Run `python main.py --phase test --model unet1024`

### Create submission
Run `python main.py --phase submission --model unet1024` 

## any questions? :jack_o_lantern:
Contact Lemonlemons (Andrew Moe) (RedHerring on Kaggle)
