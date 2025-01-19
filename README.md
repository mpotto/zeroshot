# zeroshot
Code and experiments for "A Learning Theory for Zero-Shot Prediction".

This repository contains code and experiments for "A Learning Theory for Zero-Shot Prediction". Please find instructions on software/hardware dependencies, reproducing all results from the manuscript below, and additional illustrations below.

## Abstract


## Background



## Quickstart

## Dependencies

We recommend a hardware environment has at least 32GB CPU RAM and a GPU with at least 12GB RAM for ease of use. The code runs in Python 3 with the standard Python scientific stack along with PyTorch and packages built on top of it.
These packages can be downloaded using `pip` by running:
```
pip install numpy scipy pandas matplotlib seaborn transformers
```
Next, please install PyTorch following the [installation instructions](https://pytorch.org/get-started/locally/) for your particular CUDA distribution. For example, for CUDA 11.8, run:
```
pip install torch --index-url https://download.pytorch.org/whl/cu118
```
Finally, we reply on Huggingface `transformers` (installed via `pip install transformers`), [OpenCLIP](https://github.com/mlfoundations/open_clip), and [CLIP Benchmark](https://github.com/LAION-AI/CLIP_benchmark). For the latter two, see the links for up-to-date installation information.

## Code

The files in the repo can be used for the following purposes.

**Direct Reproduction:** 

**Evaluations:** 

**Model Architecture:** Because models are loaded within the identify files (e.g. `miniclip.py`), all model definitions and base embedding models are copied in this file. 
Thus, only one file is needed to perform the custom evaluations in the previous step. The saved files refer to the "head" models. The architecture (without any OpenCLIP code) is stored in `src/multimodal_models.py`.

**Data:** The subset of ImageNet-Captions used is listed in `imagenet_captions_train_c250.csv` with a column for the filename of the ImageNet image and a column for the associated caption. Because the captions are included in the file, you can simply retrieve the images from the [ImageNet](https://www.image-net.org/download.php) dataset directly.

## References

If you found this repository useful, please consider citing the following paper.

```
@inproceedings{Liu2024TheBenefits,
  title={{The Benefits of Balance: From Information Projections to Variance Reduction}},
  author={Liu, Lang and Mehta, Ronak and Pal, Soumik and Harchaoui, Zaid},
  booktitle={NeurIPS},
  year={2024}
}
```
