# Closed-form-color-field
[NeurIPS2023] Official implementation of Reducing Shape-Radiance Ambiguity in Radiance Fields with a Closed-Form Color Estimation Method.

Qihang Fang*, Yafei Song*, Keqiang Li, and Liefeng Bo

\* Co-first author

Chinese Academy of Sciences, Institute of Automation

Alibaba Group

arXiv: https://arxiv.org/abs/2312.12726

## Introduction
A neural radiance field (NeRF) enables the synthesis of cutting-edge realistic novel view images of a 3D scene. 
It includes density and color fields to model the shape and radiance of a scene, respectively. Supervised by the 
photometric loss in an end-to-end training manner, NeRF inherently suffers from the shape-radiance ambiguity problem, 
*i.e.*, it can perfectly fit training views but does not guarantee decoupling the two fields correctly. 
To deal with this issue, existing works have incorporated prior knowledge to provide an independent supervision signal 
for the density field, including total variation loss, sparsity loss, distortion loss, *etc*. 
These losses are based on general assumptions about the density field, *e.g.*, 
it should be smooth, sparse, or compact, which are not adaptive to a specific scene. 
In this paper, we propose a more adaptive method to reduce the shape-radiance ambiguity. 
The key is a rendering method that is **only based on the density field**. 
Specifically, we first estimate the color field based on the density field and posed 
images in a closed form. Then NeRF's rendering process can proceed. 
We address the problems in estimating the color field, including occlusion and 
non-uniformly distributed views. Afterwards, it is applied to regularize NeRF's density field. 
As our regularization is guided by photometric loss, it is more adaptive compared to existing ones. 
Experimental results show that our method improves the density field of NeRF both qualitatively and quantitatively.

## Installation
We recommend using Anaconda to set up the environment:

```sh
conda env create -f environment.yml
conda activate cf_loss
```

Then install the library `svox2`, which includes a CUDA extension.
To install the main library, simply run
```sh
pip install -e . --verbose
```
in the main directory.

## Data

We have backends for NeRF Synthetic, LLFF, and DTU dataset formats, and the dataset will be auto-detected.

Please get the NeRF Synthetic and LLFF datasets from:
<https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1>
(`nerf_synthetic.zip` and `nerf_llff_data.zip`).

Please get the DTU dataset from the [NeuS](https://github.com/Totoro97/NeuS) project
(click "Data" and download `data_DTU.zip`).

## Running
In the `opt` directory, run

```sh
python opt_with_cf_loss.py <data_dir> \
  --train_dir=<ckpt_path> \
  --config=<config_file> \
  --cf_loss_ray_frac=<fraction_of_cf_rays> \
  --lambda_cf_loss=<weight_facctor_of_cf_loss>
```
`<data_dir>`: the directory storing image and camera data, e.g., `path/to/dtu_scan24` for the DTU scene 24.

`<ckpt_path>`: the directory to save training results.

`<config_file>`: `configs/dtu.json` for the DTU dataset, `configs/syn.json` for the NeRF Synthetic dataset,
and `configs/llff.json` for the LLFF dataset.

`<fraction_of_cf_rays>`: the faction of rays for regularization. We set it as 0.005. For batch size 5000, the value 0.005 means 25 rays.

`<weight_facctor_of_cf_loss>`: weight factor of the closed-form color loss. We set 10 for DTU, 0.1 for NeRF synthetic, and 0.5 for LLFF dataset.

## Citation
Cite as below if you find this repository helpful to your project:
```
@inproceedings{fang2023reducing,
  author       = {{Qihang Fang and Yafei Song} and Keqiang Li and Liefeng Bo},
  title        = {{Reducing Shape-Radiance Ambiguity in Radiance Fields with a Closed-Form Color Estimation Method}},
  booktitle    = {Advances in Neural Information Processing Systems 37: Annual Conference
                  on Neural Information Processing Systems 2023, NeurIPS 2023},
  year         = {2023}
}
```

## Acknowledgement
This repository is built on [Plenoxels](https://github.com/sxyu/svox2). Thanks for the great project.
