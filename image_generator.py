from tld.diffusion import DiffusionGenerator, DiffusionTransformer
from tld.configs import LTDConfig, DenoiserConfig, TrainConfig, DenoiserLoad
import torch
import os
import time
from tqdm import tqdm

"""
This script is used to generate 10k images from the test set used in metric calculations
"""

# create diffusion generator
# teacher denoiser config
# denoiser_cfg = DenoiserConfig(image_size = 32,
#                noise_embed_dims = 256,
#                patch_size = 2,
#                embed_dim = 768,
#                dropout = 0,
#                n_layers = 12,
#                text_emb_size = 768,
#                mlp_multiplier = 4)

# student models config
denoiser_cfg = DenoiserConfig()

denoiser_load = DenoiserLoad(**{'dtype': torch.float32,
  'file_url': '',
  'local_filename': '../state_dict_772000_full_feature_map.pth'})

cfg = LTDConfig(denoiser_cfg=denoiser_cfg, denoiser_load=denoiser_load)
diffusion_transformer = DiffusionTransformer(cfg)

# parameters for generating images
class_guidance = 6 
num_imgs = 1 
seed = 11

# iterate through test data and generate image from text
# Path to the directory containing .txt files with prompts
prompt_dir = '../prompts'
files = os.listdir(prompt_dir)
files.sort()

# for inference speed calculation
total_time = 0
time_nums = 0

for filename in tqdm(files):
    if filename.endswith('.txt'):
        # Construct the full path to the file
        filepath = os.path.join(prompt_dir, filename)
        
        # Read the prompt from the file
        with open(filepath, 'r') as file:
            prompt = file.read().strip()
        
        # Generate the image using the prompt
        start_time = time.time()
        out = diffusion_transformer.generate_image_from_text(
            prompt=prompt,
            class_guidance=class_guidance,
            num_imgs=num_imgs,
            seed=seed
        )
        # time calculations
        end_time = time.time()
        elapsed_time = end_time - start_time
        total_time += elapsed_time
        time_nums += 1

        # Save the generated image
        output_filename = f"generated_{os.path.splitext(filename)[0]}.png"
        out.save(f"../time_generated_images/{output_filename}") 

print("Finished generating images for all prompts.")

avg_time = total_time / time_nums
print(f"average inference time: {avg_time}")