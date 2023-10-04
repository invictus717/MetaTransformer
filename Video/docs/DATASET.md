# Data Preparation
We read and process the same way as [VideoMAE](https://github.com/MCG-NJU/VideoMAE/blob/main/DATASET.md), but with a different convention for the format of the data list file. We share some of our fine-tuning annotation files via Google Drive.

| dataset  | data type | train videos | validation videos | data list file |
| :------: | :-------: | :----------: | :---------------: | :------------: |
| k400 | video | 240436 | 19796 | [k400_list.zip](https://drive.google.com/file/d/11US3KptpqHsZ5K4wQLzs-OA3Y50OWtPJ/view?usp=sharing) |
| k600 | video | 366006 | 27935 | [k600_list.zip](https://drive.google.com/file/d/1kzfOEb6_va0ev5TYbLLMRB0SvSvtPl-S/view?usp=sharing) |
| k710 | video | 658340 | 66803 | [k710_list.zip](https://drive.google.com/file/d/1DdBiwG3cCJ60Rstx3-FuFlt9Q-mUCyE2/view?usp=sharing) |
| ssv2 | rawframes | 168913 | 24777 | [sthv2_list.zip](https://drive.google.com/file/d/1OtQzj1S0HjgUciB7cZa4MCDHXQ20FpZg/view?usp=sharing) |


## Pre-train Dataset
The pretrain dataset loads the data list file, and then process each line in the list. The pre-training data list file is in the following format:

for video data line:
> video_path 0 -1

for rawframes data line:
> frame_folder_path start_index total_frames

For example, the UnlabeledHybrid data list file containing data from multiple sources, in part:
```
# The path prefix 'your_path' can be specified by `--data_root ${PATH_PREFIX}` in scripts when training or inferencing.

your_path/k400/---QUuC4vJs.mp4 0 -1
your_path/k400/--VnA3ztuZg.mp4 0 -1
...
your_path/k700/-0H3T2B9PH4_000025_000035.mp4 0 -1
your_path/k700/-1IlTIWPNs4_000043_000053.mp4 0 -1
...
your_path/webvid2m/016401_016450/1017127174.mp4 0 -1
your_path/webvid2m/026551_026600/1056070034.mp4 0 -1
...
your_path/AVA/frames/clip/zlVkeKC6Ha8 9601 300
your_path/AVA/frames/clip/zlVkeKC6Ha8 9901 300
...
your_path/SomethingV2/frames/182040 1 58
your_path/SomethingV2/frames/197728 1 29
...
```
where the AVA and Something-Something data are rawframes and the rest are videos.
## Fine-tune Dataset
There are two implementations of our finetune dataset `VideoClsDataset` and `RawFrameClsDataset`, supporting video data and rawframes data, respectively. Where SSV2 uses `RawFrameClsDataset` by default and the rest of the datasets use `VideoClsDataset`.

`VideoClsDataset` loads a data list file with the following format:
> video_path label

while `RawFrameClsDataset` loads a data list file with the following format:
> frame_folder_path total_frames label

For example, video data list and rawframes data list are shown below:
```
# The path prefix 'your_path' can be specified by `--data_root ${PATH_PREFIX}` in scripts when training or inferencing.

# k400 video data validation list
your_path/k400/jf7RDuUTrsQ.mp4 325
your_path/k400/JTlatknwOrY.mp4 233
your_path/k400/NUG7kwJ-614.mp4 103
your_path/k400/y9r115bgfNk.mp4 320
your_path/k400/ZnIDviwA8CE.mp4 244
...

# ssv2 rawframes data validation list
your_path/SomethingV2/frames/74225 62 140
your_path/SomethingV2/frames/116154 51 127
your_path/SomethingV2/frames/198186 47 173
your_path/SomethingV2/frames/137878 29 99
your_path/SomethingV2/frames/151151 31 166
...
```

## Kinetics-710
We merge the training set and validation set of Kinetics-400/600/700, then remove the duplicated videos according to YouTube IDs, and finally delete the validation videos that existed in the training set. As some videos have different category names in different versions of Kinetics (referring to [k710_identical_label_merge.json](/misc/k710_identical_label_merge.json) ), we also group them together, resulting in a Kinetics dataset with 710 categories, termed Kinetics-710 (k710) or LabeledHybrid.

In the [misc](/misc/) folder, we provide the label map files for the k400, k600, k700 and k710 that we use. The k710 classification model can be simply converted to a k{400|600|700} classification model using the `/misc/label_710to{400|600|700}.json` file that we provide.
