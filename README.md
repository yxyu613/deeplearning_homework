# Stroke2Sketch: Harnessing Stroke Attributes for Training-Free Sketch Generation - ICCV 2025

[[Paper]](http://arxiv.org/abs/2510.16319) | [[Paper (ICCV)]](https://openaccess.thecvf.com/content/ICCV2025/papers/Yang_Stroke2Sketch_Harnessing_Stroke_Attributes_for_Training-Free_Sketch_Generation_ICCV_2025_paper.pdf) | [[Project Page]](https://rane7.github.io/Stroke2Sketch/)

We propose Stroke2Sketch, a framework that accurately transfers stroke attributes from a reference sketch to a content image while preserving structure and style fidelity. The top row shows reference sketches, the leftmost column displays content images, and the central and right columns illustrate our method’s precise content preservation and expressive stroke transfer.

<img src='docs/fig1.jpg' height=180>

## Setup

### Create a Conda Environment

```bash
conda env create -f environment.yaml
conda activate stroke2sketch
```

### Download Pre-trained Weights

#### StableDiffusion 
Download the StableDiffusion weights from the [CompVis organization at Hugging Face](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5).
(download the `sd-v1-5` file).

#### U2NET 
Download the u2net.pth from:
- [Google Drive](https://drive.google.com/file/d/1IG3HdpcRiDoWNookbncQjeaPN28t90yW/view) 
- [Baidu Pan](https://pan.baidu.com/share/init?surl=BYT5Ts6BxwpB8_l2sAyCkw) (提取码: chgd)

Place the model in the directory: `./U2Net_/saved_models/`

### Demo
You can use the gradio demo locally by running:
```bash
python app.py.
```

## Acknowledgements

Stroke2Sketch is highly built on [Cross-Image-Attention](https://github.com/garibida/Cross-Image-Attention), [diffusers](https://github.com/huggingface/diffusers) library, and [Edit-Friendly DDPM Inversion](https://github.com/inbarhub/DDPM_inversion).

## Citation

If you use the code and models please cite:

```
@inproceedings{yang2025stroke2sketch,
  title={Stroke2Sketch: Harnessing Stroke Attributes for Training-Free Sketch Generation},
  author={Yang, Rui and Li, Huining and Long, Yiyi and Wu, Xiaojun and He, Shengfeng},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={16545--16554},
  year={2025}
}
```


