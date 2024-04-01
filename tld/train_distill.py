#!/usr/bin/env python3

import copy
from dataclasses import asdict

import numpy as np
import torch
import torchvision
import torchvision.utils as vutils
import wandb
from accelerate import Accelerator
from diffusers import AutoencoderKL
from PIL.Image import Image
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from tld.denoiser import Denoiser
from tld.diffusion import DiffusionGenerator
from tld.configs import ModelConfig


def eval_gen(diffuser: DiffusionGenerator, labels: Tensor, img_size: int) -> Image:
    class_guidance = 4.5
    seed = 10
    out, _ = diffuser.generate(
        labels=torch.repeat_interleave(labels, 2, dim=0),
        num_imgs=16,
        class_guidance=class_guidance,
        seed=seed,
        n_iter=40,
        exponent=1,
        sharp_f=0.1,
        img_size=img_size
    )

    out = to_pil((vutils.make_grid((out + 1) / 2, nrow=8, padding=4)).float().clip(0, 1))
    out.save(f"emb_val_cfg:{class_guidance}_seed:{seed}.png")

    return out


def count_parameters(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_parameters_per_layer(model: nn.Module):
    for name, param in model.named_parameters():
        print(f"{name}: {param.numel()} parameters")


to_pil = torchvision.transforms.ToPILImage()


def update_ema(ema_model: nn.Module, model: nn.Module, alpha: float = 0.999):
    with torch.no_grad():
        for ema_param, model_param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(alpha).add_(model_param.data, alpha=1 - alpha)



def main(config: ModelConfig) -> None:
    """main train loop to be used with accelerate"""
    denoiser_config = config.denoiser_config
    train_config = config.train_config
    dataconfig = config.data_config
    teacher_embed_dim = 768

    log_with="wandb" if train_config.use_wandb else None
    accelerator = Accelerator(mixed_precision="fp16", log_with=log_with)

    accelerator.print("Loading Data:")
    latent_train_data = torch.tensor(np.load(dataconfig.latent_path), dtype=torch.float32)
    train_label_embeddings = torch.tensor(np.load(dataconfig.text_emb_path), dtype=torch.float32)
    emb_val = torch.tensor(np.load(dataconfig.val_path), dtype=torch.float32)
    dataset = TensorDataset(latent_train_data, train_label_embeddings)
    train_loader = DataLoader(dataset, batch_size=train_config.batch_size, shuffle=True)

    vae = AutoencoderKL.from_pretrained(config.vae_cfg.vae_name, torch_dtype=config.vae_cfg.vae_dtype)

    if accelerator.is_main_process:
        vae = vae.to(accelerator.device)

    adjustment_layer = nn.Linear(denoiser_config.embed_dim, teacher_embed_dim).to(accelerator.device) # used for feature distillation to match shapes
    # load teacher model based on original from:
    teacher_denoiser = Denoiser(image_size=32, noise_embed_dims=256, patch_size=2,
            embed_dim=768, dropout=0, n_layers=12)
    state_dict = torch.load('state_dict_378000.pth', map_location=accelerator.device)
    teacher_denoiser.load_state_dict(state_dict)
    teacher_denoiser = teacher_denoiser.to(accelerator.device)
    teacher_denoiser.eval()

    model = Denoiser(**asdict(denoiser_config))

    distillation_loss_fn = nn.MSELoss() # distillation loss
    feature_loss_fn = nn.MSELoss() # feature distillation loss
    reconstruction_loss_fn = nn.MSELoss() # original task loss
    optimizer = torch.optim.Adam(model.parameters(), lr=train_config.lr)

    if train_config.compile:
        accelerator.print("Compiling model:")
        model = torch.compile(model)

    if not train_config.from_scratch:
        accelerator.print("Loading Model:")
        wandb.restore(
            train_config.model_name, run_path=f"apapiu/cifar_diffusion/runs/{train_config.run_id}", replace=True
        )
        full_state_dict = torch.load(train_config.model_name)
        model.load_state_dict(full_state_dict["model_ema"])
        optimizer.load_state_dict(full_state_dict["opt_state"])
        global_step = full_state_dict["global_step"]
    else:
        global_step = 0

    if accelerator.is_local_main_process:
        ema_model = copy.deepcopy(model).to(accelerator.device)
        diffuser = DiffusionGenerator(ema_model, vae, accelerator.device, torch.float32)

    accelerator.print("model prep")
    model, train_loader, optimizer = accelerator.prepare(model, train_loader, optimizer)

    if train_config.use_wandb:
        accelerator.init_trackers(project_name="cifar_diffusion", config=asdict(config))

    accelerator.print(count_parameters(model))
    accelerator.print(count_parameters_per_layer(model))

    ### Train:
    for i in range(1, train_config.n_epoch + 1):
        accelerator.print(f"epoch: {i}")

        for x, y in tqdm(train_loader):
            x = x / config.vae_cfg.vae_scale_factor

            noise_level = torch.tensor(
                np.random.beta(train_config.beta_a, train_config.beta_b, len(x)), device=accelerator.device
            )
            signal_level = 1 - noise_level
            noise = torch.randn_like(x)

            x_noisy = noise_level.view(-1, 1, 1, 1) * noise + signal_level.view(-1, 1, 1, 1) * x

            x_noisy = x_noisy.float()
            noise_level = noise_level.float()
            label = y

            prob = 0.15
            mask = torch.rand(y.size(0), device=accelerator.device) < prob
            label[mask] = 0  # OR replacement_vector

            if global_step % train_config.save_and_eval_every_iters == 0:
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    ##eval and saving:
                    out = eval_gen(diffuser=diffuser, labels=emb_val, img_size=denoiser_config.image_size)
                    out.save("img.jpg")
                    if train_config.use_wandb:
                        accelerator.log({f"step: {global_step}": wandb.Image("img.jpg")})

                    opt_unwrapped = accelerator.unwrap_model(optimizer)
                    full_state_dict = {
                        "model_ema": ema_model.state_dict(),
                        "opt_state": opt_unwrapped.state_dict(),
                        "global_step": global_step,
                    }
                    if train_config.save_model:
                        accelerator.save(full_state_dict, train_config.model_name)
                        if train_config.use_wandb:
                            wandb.save(train_config.model_name)

            model.train()

            with accelerator.accumulate():
                ###train loop:
                optimizer.zero_grad()

                # forward pass through student
                student_pred, student_features = model(x_noisy, noise_level.view(-1, 1), label, return_features=True)
                reconstruction_loss = reconstruction_loss_fn(student_pred, x)
                # forward pass through teacher
                with torch.no_grad():
                    teacher_pred, teacher_features = teacher_denoiser(x_noisy, noise_level.view(-1, 1), label, return_features=True)
                
                feature_loss = sum(feature_loss_fn(adjustment_layer(s_feat), t_feat.detach()) for s_feat, t_feat in zip(student_features, teacher_features))
                distillation_loss = distillation_loss_fn(student_pred, teacher_pred)
                
                total_loss = reconstruction_loss + train_config.output_weight * distillation_loss + train_config.feature_weight * feature_loss

                accelerator.log({"train_loss": total_loss.item(), "reconstruction_loss": reconstruction_loss.item(), "distillation_loss": distillation_loss.item(), "feature_loss": feature_loss.item()}, step=global_step)
                accelerator.backward(total_loss)
                optimizer.step()

                if accelerator.is_main_process:
                    update_ema(ema_model, model, alpha=train_config.alpha)

            global_step += 1
    accelerator.end_training()


# args = (config, data_path, val_path)
# notebook_launcher(training_loop)
