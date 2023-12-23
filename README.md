# Closed-form-color-field
[NeurIPS2023] Official implementation of Reducing Shape-Radiance Ambiguity in Radiance Fields with a Closed-Form Color Estimation Method.

Qihang Fang*, Yafei Song*, Keqiang Li, and Liefeng Bo

\* Co-first author

Chinese Academy of Sciences, Institute of Automation

Alibaba Group

arXiv: https://arxiv.org/abs/2312.12726

A neural radiance field (NeRF) enables the synthesis of cutting-edge realistic novel view images of a 3D scene. It includes density and color fields to model the shape and radiance of a scene, respectively. Supervised by the photometric loss in an end-to-end training manner, NeRF inherently suffers from the shape-radiance ambiguity problem, *i.e.*, it can perfectly fit training views but does not guarantee decoupling the two fields correctly. To deal with this issue, existing works have incorporated prior knowledge to provide an independent supervision signal for the density field, including total variation loss, sparsity loss, distortion loss, *etc*. These losses are based on general assumptions about the density field, *e.g.*, it should be smooth, sparse, or compact, which are not adaptive to a specific scene. In this paper, we propose a more adaptive method to reduce the shape-radiance ambiguity. The key is a rendering method that is **only based on the density field**. Specifically, we first estimate the color field based on the density field and posed images in a closed form. Then NeRF's rendering process can proceed. We address the problems in estimating the color field, including occlusion and non-uniformly distributed views. Afterwards, it is applied to regularize NeRF's density field. As our regularization is guided by photometric loss, it is more adaptive compared to existing ones. Experimental results show that our method improves the density field of NeRF both qualitatively and quantitatively.

```
@inproceedings{fang2023reducing,
  author       = {{Qihang Fang and Yafei Song} and Keqiang Li and Liefeng Bo},
  title        = {{Reducing Shape-Radiance Ambiguity in Radiance Fields with a Closed-Form Color Estimation Method}},
  booktitle    = {Advances in Neural Information Processing Systems 37: Annual Conference
                  on Neural Information Processing Systems 2023, NeurIPS 2023},
  year         = {2023}
}
```
