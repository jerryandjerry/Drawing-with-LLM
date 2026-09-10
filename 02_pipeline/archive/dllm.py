# %%
import torch
from multiprocessing import Process, Manager
from io import BytesIO
from PIL import Image as PILImage

# ——— 1) Placeholders for CPU-loaded pipelines & their names ———
_pipe0 = None
_pipe1 = None
_name0 = None
_name1 = None

# ——— 2) Shared negative prompt ———
negative_prompt = (
    'text, logo, mirror reflection, high-reflective, lines, deformed, ugly, '
    'wrong proportion, low res, bad anatomy, worst quality, low quality, '
    'framing, hatching, patterns, outlines'
)

# ——— 3) Offload toggle for all three models ———
offload_models = {}  # {'flash':bool,'sdxl':bool,'flux':bool}

# job → per‐rank attempts mapping
_job_attempts = {}

# ——— 4) Generation configs dict ———
_gen_configs = {}

def set_gen_configs(configs: dict):
    """
    configs: { model_name: {width:…, height:…, ...}, … }
    """
    global _gen_configs
    _gen_configs = configs

def set_pipelines(pipe0, name0, pipe1, name1):
    """
    Inject the two CPU-loaded pipelines and their model names.
    """
    global _pipe0, _pipe1, _name0, _name1
    _pipe0, _name0 = pipe0, name0
    _pipe1, _name1 = pipe1, name1

def set_offload_models(config):
    global offload_models
    offload_models = config

def worker_loop(rank, in_queue, out_queue,
                pipe0, name0, pipe1, name1,
                gen_configs, offload_models):
    torch.cuda.set_device(rank)
    pipe       = pipe0 if rank == 0 else pipe1
    model_name = name0 if rank == 0 else name1

    # move main pipeline to GPU
    pipe = pipe.to(rank)

    # always move Flux submodules to GPU
    if model_name == 'flux':
        pipe.transformer.to(rank)
        pipe.text_encoder_2.to(rank)
        if hasattr(pipe, 'vae'):
            pipe.vae.to(rank)

    # compile + optional CPU offload for flash/sdxl
    if model_name in ('flash','sdxl'):
        pipe.unet = torch.compile(pipe.unet, backend="eager", fullgraph=True)
        if offload_models.get(model_name, False):
            pipe.enable_model_cpu_offload()

    # flux: optional group offload
    elif model_name == 'flux' and offload_models.get('flux', False):
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

    # main loop: receive (prompt, tag, attempts)
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
    assert _pipe0 is not None and _pipe1 is not None, "Must call set_pipelines() first"
    assert _gen_configs, "Must call set_gen_configs() first"
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
                _pipe0, _name0,
                _pipe1, _name1,
                _gen_configs,
                offload_models
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
        attempts = {0:1,1:1}
    _job_attempts[tag] = attempts
    for rank, q in enumerate(in_queues):
        text = prompt0 if rank==0 else prompt1
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
    (imgs0 if r==0 else imgs1).append(img)
    while len(imgs0)<total0 or len(imgs1)<total1:
        rr, _, bb = out_queue.get()
        i = PILImage.open(BytesIO(bb))
        (imgs0 if rr==0 else imgs1).append(i)

    del _job_attempts[tag]
    return imgs0 + imgs1

def stop_workers(in_queues, procs):
    for q in in_queues:
        q.put(None)
    for p in procs:
        p.join()



