from diffusers import AutoencoderKL, DDIMScheduler, StableDiffusionInpaintPipeline
import torch
from PIL import Image
import os

import config as cfg
from ip_adapter.ip_adapter import IPAdapter

def get_scene_ids(path):
    scene_ids = []
    for scene_id in os.listdir(path):
        if scene_id.startswith('.'):
            continue
        scene_ids.append(scene_id)

    return scene_ids

def save_images(image_list, path, filename_prefix):
    for i, image in enumerate(image_list):
        image.save(f"{path}/{filename_prefix}_{i}.png", lossless=True, quality=100)    



device = "cuda"

noise_scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    clip_sample=False,
    set_alpha_to_one=False,
    steps_offset=1,
)

vae = AutoencoderKL.from_single_file(cfg.vae_model_path).to(dtype=torch.float16)

pipe = StableDiffusionInpaintPipeline.from_single_file(cfg.sd15_base_model_path, 
                                                        torch_dtype=torch.float16,
                                                        vae=vae,
                                                        scheduler=noise_scheduler,
                                                        feature_extractor=None,
                                                        safety_checker=None)

pipe.to(device)
strength = 0.95
ip_adapter = IPAdapter(pipe, cfg.ipadapter_sd15_path, cfg.image_encoder_sd15_path, device=device)
generator = torch.Generator()
num_images_per_prompt = 4

samples_parent_dir = 'assets/Apr_12'
syn_parent_dir = '../Photoconsistent-NVS/instance-data/samples'
scene_ids = get_scene_ids(samples_parent_dir)# DBG: test
for scene_id in scene_ids:
    print(f'Processing {scene_id}...') # DBG: message
    if os.path.exists(f'{samples_parent_dir}/{scene_id}/inpainted_ref_3.png'):
        print('Already Processed, skip.')
        continue

    ref = Image.open(f"{samples_parent_dir}/{scene_id}/1.png").resize((1280, 720))
    mask = Image.open(f"{samples_parent_dir}/{scene_id}/mask.png")
    masked_target = Image.open(f"{samples_parent_dir}/{scene_id}/masked_121.png")
    syn = Image.open(f"{syn_parent_dir}/{scene_id}/samples/00000000/images/0004.png")
    
    """
    Base model with three reference images and mandelbrot negative images
    As an example I'm sending a mandelbrot as negative with often surprising results
    """
    noise = Image.effect_mandelbrot((224, 224), (-3, -2.5, 2, 2.5), 100)

    prompt_embeds, negative_prompt_embeds = ip_adapter.get_prompt_embeds(
        [ref, syn],
        prompt="real world scene",
        negative_prompt="blurry, worst quality, low quality",
    )
    ip_adapter.set_scale(1.0)

    images_both = pipe(
        image=masked_target, 
        mask_image=mask,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        num_inference_steps=30,
        num_images_per_prompt=num_images_per_prompt,
        generator=generator,
        strength=strength,
    ).images
    save_images(images_both, f"{samples_parent_dir}/{scene_id}", "inpainted_both")

    prompt_embeds, negative_prompt_embeds = ip_adapter.get_prompt_embeds(
        [ref],
        prompt="real world scene",
        negative_prompt="blurry, worst quality, low quality",
    )

    images_ref = pipe(
        image=masked_target, 
        mask_image=mask,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        num_inference_steps=30,
        num_images_per_prompt=num_images_per_prompt,
        generator=generator,
        strength=strength,
    ).images
    save_images(images_ref, f"{samples_parent_dir}/{scene_id}", "inpainted_ref")
