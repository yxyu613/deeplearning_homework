import os
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')

import math
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple, Optional, Tuple, List

import numpy as np
import torch
from torch import inference_mode
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
from diffusers import DDIMScheduler

from models.stable_diffusion import CrossImageAttentionStableDiffusionPipeline
from models.unet_2d_condition import FreeUUNet2DConditionModel
import re
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import cv2

from TEED.ted import TED
from U2Net_.u2net import U2NET

try:
    import pyrallis
except ImportError:
    pyrallis = None
try:
    from diffusers.training_utils import set_seed as _hf_set_seed
except Exception:
    _hf_set_seed = None

# ---- Index constants ----
SKE_INDEX = 0
REF_INDEX = 1
CNT_INDEX = 2
CONT_INDEX = 3

# ---------------- Range & Config ----------------
class Range(NamedTuple):
    start: int
    end: int

# ---------------- BLIP2 caption ----------------
_blip2_processor = None
_blip2_model = None

def strip_color_adjectives(text: str) -> str:
    colors = [
        "red","green","blue","yellow","orange","purple","pink","black","white","gray","grey",
        "brown","beige","cyan","magenta","violet","turquoise","gold","silver"
    ]
    pattern = r"\b(" + "|".join(colors) + r")\b[-\s]*"
    cleaned = re.sub(pattern, "", text, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned

def get_blip2_caption(img: Image.Image, device: str = "cuda") -> str:
    global _blip2_processor, _blip2_model
    if _blip2_processor is None:
        _blip2_processor = Blip2Processor.from_pretrained(
            "model/blip2-flan-t5-xl",
            local_files_only=True
        )
    if _blip2_model is None:
        _blip2_model = Blip2ForConditionalGeneration.from_pretrained(
            "model/blip2-flan-t5-xl",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            local_files_only=True
        ).to(device if torch.cuda.is_available() else "cpu")
    inputs = _blip2_processor(images=img, return_tensors="pt").to(_blip2_model.device)
    with torch.no_grad():
        generated_ids = _blip2_model.generate(**inputs, max_new_tokens=40)
    caption = _blip2_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return caption.strip()

def make_content_prompt(img: Image.Image, device: str) -> str:
    caption = get_blip2_caption(img, device=device)
    caption = strip_color_adjectives(caption)
    return f"Sketch of {caption}"

@dataclass
class RunConfig:
    ref_path: Path
    cnt_path: Path
    object_prompt: Optional[str] = None
    output_path: Path = Path('./output')
    seed: int = 42
    prompt: Optional[str] = None
    num_timesteps: int = 100
    cross_attn_64_range: Range = Range(10, 90)
    cross_attn_32_range: Range = Range(10, 70)
    adain_range: Range = Range(20, 100)
    swap_guidance_scale: float = 3.5
    contrast_strength: float = 1.67
    object_noun: Optional[str] = None
    load_latents: bool = True
    skip_steps: int = 30
    apply_mask: bool = False
    edge_method: str = 'teed'
    edge_alpha: float = 0.55
    edge_save_path: Optional[Path] = None
    device: str = 'cuda'
    contour_gamma: float = 1.0
    gamma: float = 0.2
    temperature: float = 1.5
    guidance_scale: float = 7.5

    def __post_init__(self):
        if isinstance(self.output_path, str):
            self.output_path = Path(self.output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        common_latents_path = self.output_path / "latents"
        common_latents_path.mkdir(parents=True, exist_ok=True)
        self.latents_path = common_latents_path
        self.ref_latent_save_path = self.latents_path / f"{self.ref_path.stem}.pt"
        self.cnt_latent_save_path = self.latents_path / f"{self.cnt_path.stem}.pt"
        self.cont_latent_save_path = self.latents_path / f"{self.cnt_path.stem}_cont.pt"

        if self.object_prompt is None:
            self.object_prompt = "object"
        if self.prompt is None:
            self.prompt = f"A sketch of a {self.object_prompt}"
        if self.object_noun is None:
            self.object_noun = self.object_prompt

        if self.device == 'cuda' and not torch.cuda.is_available():
            self.device = 'cpu'

def get_stable_diffusion_model() -> CrossImageAttentionStableDiffusionPipeline:
    print("Loading Stable Diffusion model...")
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    pipe = CrossImageAttentionStableDiffusionPipeline.from_pretrained(
        "/home/ps/project/hub/stable-diffusion-v1-5",
        safety_checker=None
    ).to(device)
    pipe.unet = FreeUUNet2DConditionModel.from_pretrained(
        "/home/ps/project/hub/stable-diffusion-v1-5", subfolder="unet"
    ).to(device)
    pipe.scheduler = DDIMScheduler.from_config(
        "/home/ps/project/hub/stable-diffusion-v1-5", subfolder="scheduler"
    )
    print("Done.")
    return pipe

# ---------------- AdaIN ----------------
def adain(cnt_feat, ref_feat):
    assert cnt_feat.size()[:2] == ref_feat.size()[:2]
    size = cnt_feat.size()
    ref_mean, ref_std = calc_mean_std(ref_feat)
    cnt_mean, cnt_std = calc_mean_std(cnt_feat)
    normalized_feat = (cnt_feat - cnt_mean.expand(size)) / cnt_std.expand(size)
    return normalized_feat * ref_std.expand(size) + ref_mean.expand(size)

def calc_mean_std(feat, eps=1e-5, mask=None):
    size = feat.size()
    if len(size) == 2:
        return calc_mean_std_2d(feat, eps, mask)
    assert len(size) == 3
    C = size[0]
    if mask is not None:
        sel = feat.view(C, -1)[:, mask.view(-1) == 1]
        feat_var = sel.var(dim=1) + eps
        feat_mean = sel.mean(dim=1).view(C, 1, 1)
    else:
        feat_var = feat.view(C, -1).var(dim=1) + eps
        feat_mean = feat.view(C, -1).mean(dim=1).view(C, 1, 1)
    feat_std = feat_var.sqrt().view(C, 1, 1)
    return feat_mean, feat_std

def calc_mean_std_2d(feat, eps=1e-5, mask=None):
    size = feat.size()
    assert len(size) == 2
    C = size[0]
    if mask is not None:
        sel = feat.view(C, -1)[:, mask.view(-1) == 1]
        feat_var = sel.var(dim=1) + eps
        feat_mean = sel.mean(dim=1).view(C, 1)
    else:
        feat_var = feat.view(C, -1).var(dim=1) + eps
        feat_mean = feat.view(C, -1).mean(dim=1).view(C, 1)
    feat_std = feat_var.sqrt().view(C, 1)
    return feat_mean, feat_std

# ---------------- Attention utilities ----------------
def enhance_tensor(tensor: torch.Tensor, contrast_factor: float = 1.67) -> torch.Tensor:
    return (tensor - tensor.mean(dim=-1)) * contrast_factor + tensor.mean(dim=-1)

def compute_scaled_dot_product_attention(Q, K, V, edit_map=False, is_cross=False, contrast_strength=1.0):
    attn_weight = torch.softmax((Q @ K.transpose(-2, -1) / math.sqrt(Q.size(-1))), dim=-1)
    if edit_map and not is_cross:
        attn_weight[SKE_INDEX] = torch.stack([
            torch.clip(enhance_tensor(attn_weight[SKE_INDEX][h], contrast_strength), 0.0, 1.0)
            for h in range(attn_weight.shape[1])
        ])
    return attn_weight @ V, attn_weight

def should_mix_keys_and_values(model, hidden_states: torch.Tensor) -> bool:
    is_in_32_timestep_range = (model.config.cross_attn_32_range.start <= model.step < model.config.cross_attn_32_range.end)
    is_in_64_timestep_range = (model.config.cross_attn_64_range.start <= model.step < model.config.cross_attn_64_range.end)
    is_hidden_states_32_square = (hidden_states.shape[1] == 32 ** 2)
    is_hidden_states_64_square = (hidden_states.shape[1] == 64 ** 2)
    should_mix = (is_in_32_timestep_range and is_hidden_states_32_square) or (is_in_64_timestep_range and is_hidden_states_64_square)
    return should_mix

# ---------------- Inversion (DDPM) ----------------
LOW_RESOURCE = True

def invert(x0, pipe, prompt_src="", num_diffusion_steps=100, cfg_scale_src=3.5, eta=1):
    pipe.scheduler.set_timesteps(num_diffusion_steps)
    with inference_mode():
        w0 = (pipe.vae.encode(x0).latent_dist.mode() * 0.18215).float()
    _, zs, wts = inversion_forward_process(pipe, w0, etas=eta, prompt=prompt_src,
                                           cfg_scale=cfg_scale_src, prog_bar=True,
                                           num_inference_steps=num_diffusion_steps)
    return zs, wts

def inversion_forward_process(model, x0, etas=None, prog_bar=False,
                              prompt="", cfg_scale=3.5,
                              num_inference_steps=50, eps=None):
    text_embeddings = encode_text(model, prompt) if prompt else None
    uncond_embedding = encode_text(model, "")
    timesteps = model.scheduler.timesteps.to(model.device)
    variance_noise_shape = (num_inference_steps,
                            model.unet.in_channels,
                            model.unet.sample_size,
                            model.unet.sample_size)
    if etas is None or (isinstance(etas, (int, float)) and etas == 0):
        eta_is_zero = True
        zs = None
    else:
        eta_is_zero = False
        if isinstance(etas, (int, float)):
            etas = [etas] * model.scheduler.num_inference_steps
        xts = sample_xts_from_x0(model, x0, num_inference_steps)
        alpha_bar = model.scheduler.alphas_cumprod
        zs = torch.zeros(size=variance_noise_shape, device=model.device)

    t_to_idx = {int(v): k for k, v in enumerate(timesteps)}
    xt = x0
    iterator = reversed(timesteps)
    if prog_bar:
        from tqdm import tqdm
        iterator = tqdm(iterator)

    for t in iterator:
        idx = t_to_idx[int(t)]
        if not eta_is_zero:
            xt = xts[idx][None]
        with torch.no_grad():
            uncond = model.unet.forward(xt, timestep=t, encoder_hidden_states=uncond_embedding)
            if text_embeddings is not None:
                cond = model.unet.forward(xt, timestep=t, encoder_hidden_states=text_embeddings)
        noise_pred = uncond.sample if text_embeddings is None else uncond.sample + cfg_scale * (cond.sample - uncond.sample)

        if eta_is_zero:
            xt = forward_step(model, noise_pred, t, xt)
        else:
            xtm1 = xts[idx + 1][None]
            alpha_bar = model.scheduler.alphas_cumprod
            pred_x0 = (xt - (1 - alpha_bar[t]) ** 0.5 * noise_pred) / alpha_bar[t] ** 0.5
            prev_t = t - model.scheduler.config.num_train_timesteps // model.scheduler.num_inference_steps
            alpha_prev = model.scheduler.alphas_cumprod[prev_t] if prev_t >= 0 else model.scheduler.final_alpha_cumprod
            variance = get_variance(model, t)
            pred_dir = (1 - alpha_prev - etas[idx] * variance) ** 0.5 * noise_pred
            mu_xt = alpha_prev ** 0.5 * pred_x0 + pred_dir
            z = (xtm1 - mu_xt) / (etas[idx] * variance ** 0.5)
            zs[idx] = z
            xtm1 = mu_xt + (etas[idx] * variance ** 0.5) * z
            xts[idx + 1] = xtm1

    if zs is not None:
        zs[-1] = torch.zeros_like(zs[-1])
    return xt, zs, xts

def encode_text(model, prompts):
    ti = model.tokenizer(prompts, padding="max_length",
                         max_length=model.tokenizer.model_max_length,
                         truncation=True, return_tensors="pt")
    with torch.no_grad():
        return model.text_encoder(ti.input_ids.to(model.device))[0]

def sample_xts_from_x0(model, x0, num_inference_steps=50):
    alpha_bar = model.scheduler.alphas_cumprod
    sqrt_one_minus = (1 - alpha_bar) ** 0.5
    timesteps = model.scheduler.timesteps.to(model.device)
    t_to_idx = {int(v): k for k, v in enumerate(timesteps)}
    shape = (num_inference_steps,
             model.unet.in_channels,
             model.unet.sample_size,
             model.unet.sample_size)
    xts = torch.zeros(shape, device=x0.device)
    for t in reversed(timesteps):
        idx = t_to_idx[int(t)]
        xts[idx] = x0 * alpha_bar[t] ** 0.5 + torch.randn_like(x0) * sqrt_one_minus[t]
    xts = torch.cat([xts, x0], dim=0)
    return xts

def forward_step(model, model_output, timestep, sample):
    next_t = min(model.scheduler.config.num_train_timesteps - 2,
                 timestep + model.scheduler.config.num_train_timesteps // model.scheduler.num_inference_steps)
    alpha_t = model.scheduler.alphas_cumprod[timestep]
    beta_t = 1 - alpha_t
    pred_x0 = (sample - beta_t ** 0.5 * model_output) / alpha_t ** 0.5
    next_sample = model.scheduler.add_noise(pred_x0, model_output, torch.LongTensor([next_t]))
    return next_sample

def get_variance(model, timestep):
    prev_t = timestep - model.scheduler.config.num_train_timesteps // model.scheduler.num_inference_steps
    alpha_t = model.scheduler.alphas_cumprod[timestep]
    alpha_prev = model.scheduler.alphas_cumprod[prev_t] if prev_t >= 0 else model.scheduler.final_alpha_cumprod
    beta_t = 1 - alpha_t
    beta_prev = 1 - alpha_prev
    variance = (beta_prev / beta_t) * (1 - alpha_t / alpha_prev)
    return variance

# ---------------- Edge & Mask ----------------
def initialize_ted(device):
    model = TED()
    ckpt = 'TEED/checkpoints/BIPED/7/7_model.pth'
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.to(device).eval()
    return model

def apply_edge_detection(image: Image.Image, cfg: RunConfig) -> Image.Image:
    if cfg.edge_method == 'teed':
        e_net = initialize_ted(cfg.device)
        t = transforms.ToTensor()(image).unsqueeze(0).to(cfg.device)
        mean_rgb = torch.tensor([123.68, 116.779, 103.939]).view(1, 3, 1, 1).to(cfg.device)
        t = t * 255 - mean_rgb
        t = t[:, [2, 1, 0], :, :]
        with torch.no_grad():
            preds = e_net(t)[-1][0, 0]
            pred = torch.sigmoid(preds).cpu().numpy()
            pred = (pred > cfg.edge_alpha).astype(np.uint8)
            pred = 1 - pred
            image = Image.fromarray(pred * 255).convert('L')
    else:
        arr = np.array(image)
        edges = cv2.Canny(arr, 100, 200)
        edges = 255 - edges
        image = Image.fromarray(edges).convert('L')
    if image.mode == 'L':
        image = image.convert('RGB')
    return image

def load_size_with_padding(image_path: Path, size: int = 512) -> Image.Image:
    img = Image.open(str(image_path)).convert('RGB')
    w, h = img.size
    scale = size / max(w, h)
    nw, nh = int(w * scale), int(h * scale)
    img = img.resize((nw, nh), Image.LANCZOS)
    canvas = Image.new('RGB', (size, size), (255, 255, 255))
    canvas.paste(img, ((size - nw) // 2, (size - nh) // 2))
    return canvas

def get_mask_u2net(image: Image.Image, cfg: RunConfig):
    if not cfg.apply_mask:
        return Image.new('L', image.size, 255)
    device = torch.device(cfg.device)
    net = U2NET().to(device)
    net.load_state_dict(torch.load('U2Net_/saved_models/u2net.pth', map_location=device))
    net.eval()
    tfms = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229,0.224,0.225])
    ])
    tensor = tfms(image).unsqueeze(0).to(device)
    with torch.no_grad():
        mask = net(tensor)[0]
    mask = (mask.squeeze().cpu() > 0.5).numpy().astype(np.uint8) * 255
    m_img = Image.fromarray(mask).resize(image.size, Image.BILINEAR)
    return m_img

def apply_mask(image: Image.Image, mask: Image.Image):
    out = Image.new("RGB", image.size, "WHITE")
    out.paste(image, mask=mask)
    return out

def load_images(cfg: RunConfig) -> Tuple[Image.Image, Image.Image, Image.Image]:
    ref = load_size_with_padding(cfg.ref_path)
    cnt = load_size_with_padding(cfg.cnt_path)
    mask = get_mask_u2net(cnt, cfg)
    cnt_masked = apply_mask(cnt, mask)
    cont = apply_edge_detection(cnt_masked, cfg)
    ref.save(cfg.output_path / 'in_ref.png')
    cnt.save(cfg.output_path / 'in_cnt.png')
    cnt_masked.save(cfg.output_path / 'masked_cnt.png')
    cont.save(cfg.output_path / 'in_cont.png')
    return ref, cnt, cont

# ---------------- Latent utilities ----------------
def load_latents_or_invert_images(model, cfg: RunConfig):
    paths_exist = (cfg.ref_latent_save_path.exists()
                   and cfg.cnt_latent_save_path.exists()
                   and cfg.cont_latent_save_path.exists())
    if cfg.load_latents and paths_exist:
        print("Load cached latents...")
        lat_ref = torch.load(cfg.ref_latent_save_path).to(cfg.device)
        lat_cnt = torch.load(cfg.cnt_latent_save_path).to(cfg.device)
        lat_cont = torch.load(cfg.cont_latent_save_path).to(cfg.device)
        noise_ref = torch.load(cfg.ref_latent_save_path.parent / f"{cfg.ref_path.stem}_ddpm_noise.pt").to(cfg.device)
        noise_cnt = torch.load(cfg.cnt_latent_save_path.parent / f"{cfg.cnt_path.stem}_ddpm_noise.pt").to(cfg.device)
        noise_cont = torch.load(cfg.cont_latent_save_path.parent / f"{cfg.cnt_path.stem}_cont_ddpm_noise.pt").to(cfg.device)
    else:
        print("Invert images...")
        ref_img, cnt_img, cont_img = load_images(cfg)
        model.enable_edit = False
        lat_ref, lat_cnt, lat_cont, noise_ref, noise_cnt, noise_cont = invert_images(ref_img, cnt_img, cont_img, model.pipe, cfg)
        model.enable_edit = True
    return lat_ref, lat_cnt, lat_cont, noise_ref, noise_cnt, noise_cont

def invert_images(ref_image: Image.Image, cnt_image: Image.Image, cont_image: Image.Image,
                  sd_model, cfg: RunConfig):
    def prep(img):
        arr = np.array(img).astype(np.float32) / 127.5 - 1.0
        return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(cfg.device)

    zs_ref, lat_ref = invert(prep(ref_image), sd_model, prompt_src=cfg.prompt,
                             num_diffusion_steps=cfg.num_timesteps, cfg_scale_src=3.5)
    zs_cnt, lat_cnt = invert(prep(cnt_image), sd_model, prompt_src=cfg.prompt,
                             num_diffusion_steps=cfg.num_timesteps, cfg_scale_src=3.5)
    zs_cont, lat_cont = invert(prep(cont_image), sd_model, prompt_src=cfg.prompt,
                               num_diffusion_steps=cfg.num_timesteps, cfg_scale_src=3.5)

    torch.save(lat_ref, cfg.ref_latent_save_path)
    torch.save(lat_cnt, cfg.cnt_latent_save_path)
    torch.save(lat_cont, cfg.cont_latent_save_path)
    torch.save(zs_ref, cfg.latents_path / f"{cfg.ref_path.stem}_ddpm_noise.pt")
    torch.save(zs_cnt, cfg.latents_path / f"{cfg.cnt_path.stem}_ddpm_noise.pt")
    torch.save(zs_cont, cfg.latents_path / f"{cfg.cnt_path.stem}_cont_ddpm_noise.pt")
    return lat_ref, lat_cnt, lat_cont, zs_ref, zs_cnt, zs_cont

def get_init_latents_and_noises(model, cfg: RunConfig):
    if model.latents_cnt is not None and model.latents_cnt.dim() == 4 and model.latents_cnt.shape[0] > 1:
        model.latents_cnt = model.latents_cnt[cfg.skip_steps]
        model.latents_ref = model.latents_ref[cfg.skip_steps]
        model.latents_cont = model.latents_cont[cfg.skip_steps]
    init_latents = torch.stack([model.latents_cnt, model.latents_ref,
                                model.latents_cnt, model.latents_cont])
    init_zs = [model.zs_cnt[cfg.skip_steps:], model.zs_ref[cfg.skip_steps:],
               model.zs_cnt[cfg.skip_steps:], model.zs_cont[cfg.skip_steps:]]
    return init_latents, init_zs

# ---------------- SketchGenerationModel ----------------
class SketchGenerationModel:
    def __init__(self, config: RunConfig, pipe=None):
        self.down_layers = []
        self.middle_layers = []
        self.up_layers = []

        self.config = config
        self.pipe = get_stable_diffusion_model() if pipe is None else pipe
        self.segmentor = None
        self.latents_ref = self.latents_cnt = self.latents_cont = None
        self.zs_ref = self.zs_cnt = self.zs_cont = None
        self.image_ref_mask_32 = self.image_cnt_mask_32 = None
        self.image_ref_mask_64 = self.image_cnt_mask_64 = None
        self.enable_edit = False
        self.step = 0

        self.register_attention_control()
        try:
            self.segmentor = __import__("utils.segmentation", fromlist=["Segmentor"]).Segmentor(prompt=config.prompt, object_nouns=[config.object_noun])
        except Exception:
            class _SimpleSeg:
                def __init__(self, prompt, object_nouns): pass
                def get_object_masks(self): 
                    return (torch.ones(1,32,32).to(config.device), torch.ones(1,32,32).to(config.device),
                            torch.ones(1,64,64).to(config.device), torch.ones(1,64,64).to(config.device))
                def update_attention(self, *a, **k): return
            self.segmentor = _SimpleSeg(config.prompt, [config.object_noun])

    def set_latents(self, latents_ref: torch.Tensor, latents_cnt: torch.Tensor, latents_cont: torch.Tensor):
        self.latents_ref = latents_ref
        self.latents_cnt = latents_cnt
        self.latents_cont = latents_cont

    def set_noise(self, zs_ref: torch.Tensor, zs_cnt: torch.Tensor, zs_cont: torch.Tensor):
        self.zs_ref = zs_ref
        self.zs_cnt = zs_cnt
        self.zs_cont = zs_cont

    def set_masks(self, masks: List[torch.Tensor]):
        self.image_ref_mask_32, self.image_cnt_mask_32, self.image_ref_mask_64, self.image_cnt_mask_64 = masks

    def get_adain_callback(self):
        def callback(st: int, timestep: int, latents: torch.FloatTensor) -> None:
            self.step = st
            if self.config.adain_range.start <= self.step < self.config.adain_range.end:
                latents[0] = adain(latents[0], latents[1])
        return callback

    def register_attention_control(self):
        model_self = self

        class AttentionProcessor:
            def __init__(self, place_in_unet: str, query_preserve=False, layer_name=""):
                self.place_in_unet = place_in_unet
                self.query_preserve = query_preserve
                self.layer_name = layer_name

            def __call__(self, attn, hidden_states: torch.Tensor, encoder_hidden_states: Optional[torch.Tensor] = None,
                         attention_mask=None, temb=None, perform_swap: bool = False):
                residual = hidden_states
                if hasattr(attn, "spatial_norm") and attn.spatial_norm is not None:
                    hidden_states = attn.spatial_norm(hidden_states, temb)
                input_ndim = hidden_states.ndim
                if input_ndim == 5:
                    batch_size, channel, height, width = hidden_states.shape
                    hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
                batch_size, sequence_length, _ = (hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape)
                if attention_mask is not None:
                    attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
                    attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])
                if hasattr(attn, "group_norm") and attn.group_norm is not None:
                    hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
                query = attn.to_q(hidden_states)
                is_cross = encoder_hidden_states is not None
                if not is_cross:
                    encoder_hidden_states = hidden_states
                elif hasattr(attn, "norm_cross") and attn.norm_cross:
                    encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
                key = attn.to_k(encoder_hidden_states)
                value = attn.to_v(encoder_hidden_states)
                inner_dim = key.shape[-1]
                head_dim = inner_dim // attn.heads
                should_mix = False
                vis_flag = False
                if perform_swap and not is_cross and "up" in self.place_in_unet and model_self.enable_edit:
                    if should_mix_keys_and_values(model_self, hidden_states):
                        should_mix = True
                        if model_self.step % 10 == 0 and model_self.step < 60:
                            key[SKE_INDEX] = key[CNT_INDEX] + key[CONT_INDEX]
                            value[SKE_INDEX] = value[CNT_INDEX] + value[CONT_INDEX]
                        else:
                            key[SKE_INDEX] = key[REF_INDEX]
                            value[SKE_INDEX] = value[REF_INDEX]
                        if self.query_preserve:
                            vis_flag = True
                            query[SKE_INDEX] = query[CONT_INDEX] * model_self.config.gamma + query[SKE_INDEX] * (1 - model_self.config.gamma)
                query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
                key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
                value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
                hidden_states, attn_weight = compute_scaled_dot_product_attention(
                    query, key, value,
                    edit_map=perform_swap and model_self.enable_edit and should_mix,
                    is_cross=is_cross,
                    contrast_strength=model_self.config.contrast_strength,
                )
                try:
                    model_self.segmentor.update_attention(attn_weight, is_cross)
                except Exception:
                    pass
                hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
                hidden_states = hidden_states.to(query[SKE_INDEX].dtype)
                if isinstance(attn.to_out, torch.nn.ModuleList) and len(attn.to_out) >= 2:
                    hidden_states = attn.to_out[0](hidden_states)
                    hidden_states = attn.to_out[1](hidden_states)
                else:
                    try:
                        hidden_states = attn.to_out(hidden_states)
                    except Exception:
                        pass
                if input_ndim == 5:
                    hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
                if getattr(attn, "residual_connection", False):
                    hidden_states = hidden_states + residual
                hidden_states = hidden_states / getattr(attn, "rescale_output_factor", 1.0)
                return hidden_states

        def register_recr(net_, name, count, place_in_unet):
            if net_.__class__.__name__ == 'ResnetBlock2D':
                pass
            if net_.__class__.__name__ == 'Attention':
                if name.endswith("attn1"):
                    if place_in_unet == "down":
                        model_self.down_layers.append(name)
                    elif place_in_unet == "mid":
                        model_self.middle_layers.append(name)
                    elif place_in_unet == "up":
                        model_self.up_layers.append(name)
                qp = len(model_self.up_layers) >= 4
                try:
                    net_.set_processor(AttentionProcessor(place_in_unet + f"_{count + 1}", qp))
                except Exception:
                    pass
                return count + 1
            elif hasattr(net_, 'named_children'):
                for child_name, net__ in net_.named_children():
                    new_full_name = f"{name}.{child_name}" if name else child_name
                    count = register_recr(net__, new_full_name, count, place_in_unet)
            return count

        cross_att_count = 0
        try:
            sub_nets = self.pipe.unet.named_children()
            for net_name, net in sub_nets:
                if "down" in net_name:
                    cross_att_count += register_recr(net, net_name, 0, "down")
                elif "up" in net_name:
                    cross_att_count += register_recr(net, net_name, 0, "up")
                elif "mid" in net_name:
                    cross_att_count += register_recr(net, net_name, 0, "mid")
        except Exception:
            pass

# ---------------- Core run ----------------
def _seed_all(seed: int):
    if _hf_set_seed:
        _hf_set_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

def run(cfg: RunConfig) -> List[Image.Image]:
    _seed_all(cfg.seed)
    model = SketchGenerationModel(cfg)
    lat_ref, lat_cnt, lat_cont, noise_ref, noise_cnt, noise_cont = load_latents_or_invert_images(model, cfg)
    model.set_latents(lat_ref, lat_cnt, lat_cont)
    model.set_noise(noise_ref, noise_cnt, noise_cont)
    init_latents, init_zs = get_init_latents_and_noises(model, cfg)
    model.pipe.scheduler.set_timesteps(cfg.num_timesteps)
    model.enable_edit = True
    start_step = min(cfg.cross_attn_32_range.start, cfg.cross_attn_64_range.start)
    end_step = max(cfg.cross_attn_32_range.end, cfg.cross_attn_64_range.end)
    images = model.pipe(
        prompt=[cfg.prompt] * 4,
        latents=init_latents,
        guidance_scale=1.0,
        num_inference_steps=cfg.num_timesteps,
        swap_guidance_scale=cfg.swap_guidance_scale,
        callback=model.get_adain_callback(),
        eta=1,
        zs=init_zs,
        generator=torch.Generator(device='cuda' if cfg.device.startswith('cuda') and torch.cuda.is_available() else 'cpu').manual_seed(cfg.seed),
        cross_image_attention_range=Range(start=start_step, end=end_step),
    ).images
    names = ["out_ske", "out_ref", "out_cnt", "out_cont"]
    for i, nm in enumerate(names):
        images[i].save(cfg.output_path / f"{nm}-skip_steps_{cfg.skip_steps}_gamma_{cfg.gamma}-SGS_{cfg.swap_guidance_scale}-CS_{cfg.contrast_strength}.png")
    joined = np.concatenate(images[::-1], axis=1)
    Image.fromarray(joined).save(cfg.output_path / f"out_joined-skip_steps_{cfg.skip_steps}_gamma_{cfg.gamma}-SGS_{cfg.swap_guidance_scale}-CS_{cfg.contrast_strength}.png")
    return images

# ---------------- Gradio interface ----------------
def extract_and_display_ske_cnt(col_image, ref_image,
                                skip_steps, gamma, edge_method, edge_alpha,
                                swap_guidance_scale, contrast_strength,
                                load_latents, apply_mask):
    save_dir = Path('./output')
    save_dir.mkdir(exist_ok=True)
    col_path = save_dir / 'col_input.jpg'
    ref_path = save_dir / 'ref_input.jpg'
    col_image.save(col_path)
    ref_image.save(ref_path)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #dynamic_prompt = make_content_prompt(col_image, device=device)
    cfg = RunConfig(
        ref_path=ref_path,
        cnt_path=col_path,
        output_path=save_dir,
        skip_steps=skip_steps,
        gamma=gamma,
        edge_method=edge_method,
        edge_alpha=edge_alpha,
        swap_guidance_scale=swap_guidance_scale,
        contrast_strength=contrast_strength,
        load_latents=load_latents,
        apply_mask=apply_mask,
        device=device,
        #prompt=dynamic_prompt,
        #object_prompt=None
    )
    imgs = run(cfg)
    return imgs[0].resize((512, 512))#, imgs[3].resize((512, 512))


if __name__ == "__main__":
    if pyrallis is not None:
        @pyrallis.wrap()
        def main(cfg: RunConfig):
            run(cfg)
        main()
    else:
        cfg = RunConfig(
            ref_path=Path('./input/ref.png'),
            cnt_path=Path('./input/cnt.png'),
            output_path=Path('./output'),
            seed=42,
            prompt=None,
            num_timesteps=100,
            cross_attn_64_range=Range(10, 90),
            cross_attn_32_range=Range(10, 70),
            adain_range=Range(20, 100),
            swap_guidance_scale=3.5,
            contrast_strength=1.67,
            object_noun=None,
            load_latents=True,
            skip_steps=30,
            apply_mask=False,
            edge_method='teed',
            edge_alpha=0.55,
            edge_save_path=None,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            contour_gamma=1.0,
            gamma=0.25,
            temperature=1.5,
            guidance_scale=7.5
        )
        run(cfg)
