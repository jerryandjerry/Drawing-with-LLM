import torch, gc
from multiprocessing import Process, Manager
from io import BytesIO
from PIL import Image as PILImage
from diffusers import DiffusionPipeline, DPMSolverSinglestepScheduler, DDIMScheduler
from diffusers import FluxTransformer2DModel
from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig, FluxTransformer2DModel, FluxPipeline, GGUFQuantizationConfig
from transformers import BitsAndBytesConfig as BitsAndBytesConfig, T5EncoderModel
from diffusers import AutoencoderKL, AutoencoderTiny
from diffusers.hooks import apply_group_offloading


# ——— 1) Placeholders for model paths & their names ———
_path0 = None
_path1 = None
_name0 = None
_name1 = None

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
offload_models = {}    # {'flash':bool,'sdxl':bool,'flux':bool}
compile_models = {}    # {'flash':bool,'sdxl':bool}

# job → per‐rank attempts mapping
_job_attempts = {}

# ——— 4) Generation configs dict ———
_gen_configs = {}

def set_gen_configs(configs: dict):
    global _gen_configs
    _gen_configs = configs

def set_model_paths(path0, name0, path1, name1):
    global _path0, _path1, _name0, _name1
    _path0, _name0 = path0, name0
    _path1, _name1 = path1, name1

def set_offload_models(config: dict):
    global offload_models
    offload_models = config

def set_compile_models(config: dict):
    global compile_models
    compile_models = config

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
    base.load_lora_weights(_hypersd_path, weight_name="Hyper-SDXL-12steps-CFG-lora.safetensors")
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

def worker_loop(rank, in_queue, out_queue,
                path0, name0, path1, name1,
                gen_configs, offload_models, compile_models,
                lora_path, hypersd_path, t5_path, flux_path, vae_path):
    torch.cuda.set_device(rank)

    # re-bind your aux paths in the child
    global _lora_path, _hypersd_path, _t5_path, _flux_path, _vae_path
    _lora_path, _hypersd_path, _t5_path, _flux_path, _vae_path = (
        lora_path, hypersd_path, t5_path, flux_path, vae_path
    )

    model_path = path0    if rank == 0 else path1
    model_name = name0    if rank == 0 else name1

    # load & CPU‐instantiate exactly once
    pipe = _loaders[model_name](model_path)

    # move to GPU
    pipe = pipe.to(rank)

    # Flux submodules → GPU
    if model_name == 'flux':
        pipe.transformer.to(rank)
        pipe.text_encoder_2.to(rank)
        if hasattr(pipe, 'vae'):
            pipe.vae.to(rank)

    # flash/sdxl: optional compile + CPU offload
    if model_name in ('flash','sdxl'):
        if compile_models.get(model_name, False):
            pipe.unet = torch.compile(pipe.unet, backend="eager", fullgraph=True)
        if offload_models.get(model_name, False) and rank == 0:
            pipe.enable_model_cpu_offload()

    # flux: optional group offload
    elif model_name=='flux' and offload_models.get('flux', False):
        from diffusers.hooks import apply_group_offloading
        apply_group_offloading(
            pipe.text_encoder_2,
            offload_device=torch.device('cpu'),
            onload_device=torch.device(f'cuda:{rank}'),
            offload_type='leaf_level',
            use_stream=True,
            record_stream=True
        )

    torch.cuda.empty_cache()

    # main inference loop
    while True:
        item = in_queue.get()
        if item is None:
            break
        prompt, tag, attempts = item
        local_attempts = attempts.get(rank, 1)
        cfg = gen_configs[model_name]
        for _ in range(local_attempts):
            imgs = pipe(prompt, negative_prompt=negative_prompt, **cfg).images
            for img in imgs:
                buf = BytesIO()
                img.save(buf, format="PNG")
                out_queue.put((rank, tag, buf.getvalue()))

def start_workers(world_size=2):
    assert _path0 is not None and _path1 is not None, "Must call set_model_paths() first"
    assert _gen_configs,       "Must call set_gen_configs() first"
    mgr       = Manager()
    in_queues = [mgr.Queue() for _ in range(world_size)]
    out_queue = mgr.Queue()
    procs     = []
    for r in range(world_size):
        p = Process(
            target=worker_loop,
            args=(
                r,
                in_queues[r],
                out_queue,
                _path0, _name0,
                _path1, _name1,
                _gen_configs,
                offload_models,
                compile_models,
                _lora_path,
                _hypersd_path,
                _t5_path,
                _flux_path,
                _vae_path,
            ),
            daemon=True
        )
        p.start()
        procs.append(p)
    return in_queues, out_queue, procs

def submit_to_all(prompt0, prompt1, in_queues, tag=None, attempts=None):
    import time
    if tag is None:
        tag = str(int(time.time()*1000))
    if attempts is None:
        attempts = {0:1, 1:1}
    _job_attempts[tag] = attempts
    for rank, q in enumerate(in_queues):
        text = prompt0 if rank == 0 else prompt1
        q.put((text, tag, attempts))
    return tag

def get_results(out_queue, world_size):
    r, tag, b = out_queue.get()
    attempts = _job_attempts[tag]
    cfg0 = _gen_configs[_name0]
    cfg1 = _gen_configs[_name1]
    total0 = attempts.get(0,1) * cfg0['num_images_per_prompt']
    total1 = attempts.get(1,1) * cfg1['num_images_per_prompt']

    imgs0, imgs1 = [], []
    img = PILImage.open(BytesIO(b))
    (imgs0 if r == 0 else imgs1).append(img)
    while len(imgs0) < total0 or len(imgs1) < total1:
        rr, _, bb = out_queue.get()
        i = PILImage.open(BytesIO(bb))
        (imgs0 if rr == 0 else imgs1).append(i)

    del _job_attempts[tag]
    return imgs0 + imgs1

def stop_workers(in_queues, procs):
    for q in in_queues:
        q.put(None)
    for p in procs:
        p.join()
