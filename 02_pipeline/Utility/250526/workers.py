#| export
import kagglehub
import os, io, ast, re, gc, time, math, string, statistics
from io import BytesIO
from multiprocessing import Process, Manager
import numpy as np, pandas as pd, matplotlib.pyplot as plt, cv2
import torch, torch.nn as nn
from PIL import Image as PILImage, ImageFilter
from more_itertools import chunked
import cairosvg, vtracer, clip

from transformers import AutoProcessor, PaliGemmaForConditionalGeneration, T5EncoderModel, BitsAndBytesConfig
from diffusers import (
    DiffusionPipeline, DPMSolverSinglestepScheduler, DDIMScheduler,
    AutoencoderKL, AutoencoderTiny, FluxTransformer2DModel, FluxPipeline,
    BitsAndBytesConfig as DiffusersBitsAndBytesConfig, GGUFQuantizationConfig
)
from diffusers.hooks import apply_group_offloading

from dllm_utils import map_score_to_range, compute_color_richness_entropy, bitmap_to_svg_layered
from metricMP import VQAEvaluator, AestheticEvaluator, AestheticPredictor
import metricMP as metric

input_path = "output.png"
output_path = "output.svg"

negative_prompt = (
    "text, logo, mirror reflection, high-reflective, lines, deformed, ugly, "
    "wrong proportion, low res, bad anatomy, worst quality, low quality, "
    "framing, hatching, patterns, outlines"
)

# ============================= diffusion function ==================================
# ——— 1) Placeholders for model paths & their names ———
_path0 = None
_name0 = None

# ——— 1b) Placeholders for aux checkpoints needed by your loaders ———
_lora_path    = None
_hypersd_path = None
_t5_path      = None
_flux_path    = None
_vae_path     = None

# ——— 2) Shared negative prompt ———
negative_prompt = (
    'text, logo, mirror reflection, high-reflective, lines, deformed, ugly, '
    'wrong proportion, low res, bad anatomy, worst quality, low quality, '
    'framing, hatching, patterns, outlines'
)

# ——— 3) Toggles ———
# offload_models = {}    # {'flash':bool,'sdxl':bool,'flux':bool}
# compile_models = {}    # {'flash':bool,'sdxl':bool}

# job → per‐rank attempts mapping
_job_attempts = {}

# ——— 4) Generation configs dict ———
_gen_configs = {}

def set_gen_configs(configs: dict):
    global _gen_configs
    _gen_configs = configs

def set_model_paths(path0, name0):
    global _path0,  _name0
    _path0, _name0 = path0, name0

# def set_offload_models(config: dict):
#     global offload_models
#     offload_models = config

# def set_compile_models(config: dict):
#     global compile_models
#     compile_models = config

def set_aux_paths(lora_path, hypersd_path, t5_path, flux_path, vae_path):
    """
    Provide the extra checkpoint paths your custom loaders need.
    """
    global _lora_path, _hypersd_path, _t5_path, _flux_path, _vae_path
    _lora_path    = lora_path
    _hypersd_path = hypersd_path
    _t5_path      = t5_path
    _flux_path    = flux_path
    _vae_path     = vae_path

# ——— 5) Your custom loader functions ———
def _load_sdxl(path):
    scheduler = DDIMScheduler.from_pretrained(
        path, subfolder="scheduler", timestep_spacing="trailing"
    )
    base = DiffusionPipeline.from_pretrained(
        path, torch_dtype=torch.float16, variant="fp16", scheduler=scheduler, use_safetensors=True
    )
    if "8steps" in str(_hypersd_path):
        lora_name = "Hyper-SDXL-8steps-CFG-lora.safetensors"
    else:
        lora_name = "Hyper-SDXL-12steps-CFG-lora.safetensors"
        
    base.load_lora_weights(_hypersd_path, weight_name=lora_name)
    base.fuse_lora()
    base.load_lora_weights(_lora_path, weight_name='Vector_illustration_XL.safetensors')
    base.fuse_lora(lora_scale=0.8)
    return base

def _load_flash(path):
    base = DiffusionPipeline.from_pretrained(
        path, torch_dtype=torch.float16, use_safetensors=True
    )
    base.scheduler = DPMSolverSinglestepScheduler.from_config(
        base.scheduler.config, timestep_spacing="trailing"
    )
    base.load_lora_weights(_lora_path, weight_name='Vector_illustration_XL.safetensors')
    base.fuse_lora(lora_scale=0.8)
    return base

def _load_flux(path):
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    transformer = FluxTransformer2DModel.from_pretrained(
        path,
        subfolder="transformer",
        quantization_config=nf4_config,
        torch_dtype=torch.bfloat16
    ).to('cpu')
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    t5_nf4 = T5EncoderModel.from_pretrained(
        _t5_path,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16
    ).to('cpu')
    vae = AutoencoderTiny.from_pretrained(_vae_path, torch_dtype=torch.bfloat16).to('cpu')
    base_flux = FluxPipeline.from_pretrained(
        pretrained_model_name_or_path=path,
        transformer=transformer,
        vae=vae,
        text_encoder_2=t5_nf4,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    ).to('cpu')
    base_flux.enable_vae_slicing()
    gc.collect(); torch.cuda.empty_cache()
    return base_flux

_loaders = {
    'sdxl':  _load_sdxl,
    'flash': _load_flash,
    'flux':  _load_flux,
}


# ============================= evaluator function ==================================
_pali_model = None
_pali_processor = None
def set_palimodel(pali_model, pali_processor):
    global _pali_model, _pali_processor
    _pali_model = pali_model
    _pali_processor = pali_processor

# ============================= helper function ==================================

def load(device):
        model_path = os.path.join(
            kagglehub.model_download('jiazhuang/sac-logos-ava1-l14-linearmse/Transformers/default/1'),
            'sac+logos+ava1-l14-linearMSE.pth',
        )
        clip_model_path = os.path.join(
            kagglehub.model_download('jiazhuang/clip-vit-large-patch14/Transformers/default/1'),
            'ViT-L-14.pt',
        )
        state_dict = torch.load(model_path, weights_only=True, map_location=device)
        
        # CLIP embedding dim is 768 for CLIP ViT L 14
        predictor = AestheticPredictor(768)
        predictor.load_state_dict(state_dict)
        predictor.to(device)
        predictor.eval()
        clip_model, preprocessor = clip.load(clip_model_path, device=device)
        
        return predictor, clip_model, preprocessor

def image_resize(image, size=(384, 384)):
    return image.convert('RGB').resize(size)

def get_score(sample, qa, aesthetic_evaluator, vqa_evaluator, vqa = True,):

    try:
        # If sample is a string, treat as SVG and convert to image
        rng = np.random.RandomState(42)
        group_seed = rng.randint(0, np.iinfo(np.int32).max)
        if isinstance(sample, str):
            image = metric.svg_to_png(sample)
        else:
            image = sample
        
        image_processor = metric.ImageProcessor(image=image_resize(image), seed=group_seed).apply()
        image = image_processor.image.copy()
    
        try:
            aesthetic_score = aesthetic_evaluator.score(image)
        except Exception as e:
            print(f"AES score error: {e}")
            aesthetic_score = 0.5

        if vqa:
            try:
                questions = qa['question']
                choices = qa['choices']
                answers = qa['answer']
                vqa_score = vqa_evaluator.score(questions, choices, answers, image)
            except Exception as e:
                raise
                # print(f"VQA score error: {e}")
                # vqa_score = 0.5
        else:
            vqa_score = 0.5
            
        ocr_score = 1.0
    
        instance_score = metric.harmonic_mean(vqa_score, aesthetic_score, beta=0.5) * ocr_score
    
        return instance_score, aesthetic_score, ocr_score, vqa_score
    
    except Exception as e:
        raise
        # print(f"score error: {e}")
        # return 0.5, 0.5, 1.0, 0.5

        
def drain_queue(q):
    try:
        while True:
            q.get_nowait()
    except:
        pass
        
# ============================= main function ==================================
def _generator_loop(path0, name0, gen_configs,
                    lora_path, hypersd_path, t5_path, flux_path, vae_path,
                    prompt_q, image_q, rank):
    # rank == 1
    torch.cuda.set_device(rank)

    global _lora_path, _hypersd_path, _t5_path, _flux_path, _vae_path
    _lora_path, _hypersd_path, _t5_path, _flux_path, _vae_path = (
        lora_path, hypersd_path, t5_path, flux_path, vae_path
    )
    # cfg       = gen_configs[model_name]
    model_path      = path0
    model_name = name0
    lora_path = lora_path

    pipe = _loaders[model_name](model_path).to(f'cuda:{rank}')
    # pipe.enable_vae_slicing()
    if model_name in ('flash','sdxl'):
        pipe.unet = torch.compile(pipe.unet, backend="eager", fullgraph=True)
    torch.cuda.empty_cache()
    
    while True:
        item = prompt_q.get()
        if item is None:
            break
        tag, prompt, qa, attempts = item
        drain_queue(image_q)
        cfg = gen_configs[model_name]
        for i in range(attempts):
            imgs = pipe(prompt, negative_prompt=negative_prompt, **cfg).images
            for bitmap in imgs:
                color_score = compute_color_richness_entropy(bitmap)
                resolution = map_score_to_range(color_score)
                # bitmap → SVG
                svg = bitmap_to_svg_layered(bitmap, input_path, output_path, resolution)
                buf = BytesIO()
                bitmap.save(buf, format="PNG")
                image_q.put((tag, buf.getvalue(), svg, qa))
            print(f'image{i} outsourced!')

def _scorer_loop(image_q, result_q, pali_model, pali_processor, rank):
    # rank == 0
    torch.cuda.set_device(rank)

    # pali_model = pali_model.to(f'cuda:{rank}')
    # create instance
    vqa_evaluator = VQAEvaluator(pali_model, pali_processor)
    
    # load aes model =============
    aes_predictor, clip_model, aes_preprocessor = load(f'cuda:{rank}')
    aesthetic_evaluator = AestheticEvaluator(aes_predictor, clip_model, aes_preprocessor, f'cuda:{rank}')
    print('evaluators loaded')
    from io import BytesIO
    while True:
        item = image_q.get()
        print('image got')
        if item is None:
            break
        tag, png_byte, svg, qa = item

        # VQA/aesthetic scoring
        instance_score, aesthetic_score, ocr_score, vqa_score = get_score(sample=svg, qa=qa, aesthetic_evaluator= aesthetic_evaluator, vqa_evaluator = vqa_evaluator)
        result_q.put((tag, instance_score, aesthetic_score, ocr_score, vqa_score, svg, png_byte))
        gc.collect(); torch.cuda.empty_cache()

def start_workers():
    mgr = Manager()
    prompt_q = mgr.Queue()
    image_q = mgr.Queue()
    result_q = mgr.Queue()

    p0 = Process(target=_scorer_loop, args=(image_q, result_q, _pali_model, _pali_processor ,0), daemon=True)
    p1 = Process(target=_generator_loop, args=(_path0, _name0, _gen_configs,
                    _lora_path, _hypersd_path, _t5_path, _flux_path, _vae_path,
                                               prompt_q, image_q, 1), daemon=True)

    p0.start()
    p1.start()

    return prompt_q, result_q, [p0, p1], image_q

def stop_workers(prompt_q, image_q, procs):
    prompt_q.put(None)
    image_q.put(None)
    for p in procs:
        p.join()
