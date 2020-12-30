# MultiModal Neural Machine Translation

# Background
This project considers the problem of image captioning is considered with aim to produce a translation in a target language given the caption in source language, the corresponding image. Attention Models are used to make use of the visual features to disambiguate the words.
This problem is a part of the [AMT workshop](http://lotus.kuee.kyoto-u.ac.jp/WAT/WAT2019/index.html). The dataset can be downloaded from [here](https://ufal.mff.cuni.cz/hindi-visual-genome).

The dataset looks like this-
![Images.](https://github.com/tejasvi96/Neural-Machine_Translation/blob/main/images/multimodal.png?raw=true)

We evaluate the models on RIBES score and BLEU scores.

# Experiments

Two approaches were considered.

1. Using text only features making use of a sequence to sequence attention based model.
1. Making use of the image features as well where we can use image feature by extracting the features output from a CNN model like VGG 's fc7 layer and convert this to the word embedding space using a separate network. Other approaches involve making use of the image features by downsampling it to the hidden dimension of recurrent model and using it as the first input hiddden state.

## Results

| Approach  | Bleu Score |
| ------------- |:-------------:|
| Text only    | 0.28   |
| Text + Image Features      | 0.30    |


## Sample Outputs
![Images.](/images/example1.png "This is a sample image.")

![Images.](/images/example2.png "This is a sample image.")
