3D-CNN for Lip Reading Challenge 2019

(第2回機械読唇チャレンジ)
====

Overview

## Description

Implementation the baseline method by 3D Convolutional Neural Network (3D-CNN) for Lip Reading Challenge 2019 (LR-Challenge2019).
LR-Challenge2019 is a competition for lip reading technology organized by our research group.

## Dataset

The dataset of LR-Challenge2019 is SSSD, released by our research group, and can be found at [SSSD](http://www.slab.ces.kyutech.ac.jp/SSSD/index_en.html). This dataset provides pairs of lower half face ROI (LF-ROI) images and 68 facial feature points, these are collected from 72 speakers using smart device. The speech contents is 25 Japanese words: 10 digit words and 15 greeting words.

<a href="http://www.slab.ces.kyutech.ac.jp/SSSD/index_en.html" target="_blank">SSSD</a>

A sample of LF-ROI is shown below. This is a movie saying /a-ri-ga-to-u/ (thank you) in Japanese. The image size of LF-ROI is 300x300 [pixel]. The frame rate is approx. 30fps.

![demo](s010_011_007.gif)

## Demo

- training data: 25 words x 54 speakers x 10 samples = 13,500 samples
- test data: 25 words x 18 speakers x 10 samples = 4,500 samples

|epoch|train accuracy|test accuracy|
----|----|----
|100|0.662|0.602|

## Requirement

python, opencv, Keras, numpy, tqdm

## Usage
~~~
$ python 3DCNN_SSSD.py
~~~

## Author

Takeshi Saitoh

saitoh@ces.kyutech.ac.jp

Kyushu Institute of Technology, Japan
