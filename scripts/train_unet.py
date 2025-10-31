import argparse
import os
import pickle
import random
from pathlib import Path
import yaml

from vggsound import VGGSoundNet
from nisqa.src.NISQA_lib import NISQA_DIM

from PIL import Image
import hashlib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.logging import get_logger
from datasets import load_dataset, load_from_disk
from diffusers import (AutoencoderKL, DDIMScheduler, DDPMScheduler,
                       DPMSolverMultistepScheduler, EulerDiscreteScheduler,
                       UNet2DConditionModel, UNet2DModel)
from diffusers.optimization import get_scheduler
from diffusers import Mel
from diffusers.training_utils import EMAModel
from diffusers.utils.import_utils import is_xformers_available
from huggingface_hub import HfFolder, Repository, whoami
from librosa.util import normalize
from torchvision.transforms import Compose, Normalize, ToTensor
from tqdm.auto import tqdm
from dotenv import load_dotenv
load_dotenv()

from diffusers import DiffusionPipeline
import torch
import numpy as np
from typing import Optional, Union, List, Tuple

logger = get_logger(__name__)

class clMusDiff(DiffusionPipeline):
    def __init__(self, unet: UNet2DConditionModel, scheduler: Union[DDIMScheduler, DDPMScheduler, DPMSolverMultistepScheduler, EulerDiscreteScheduler], vqvae: AutoencoderKL, mel: Mel):
        super().__init__()
        self.register_modules(
            unet=unet,
            scheduler=scheduler,
            vqvae=vqvae,
            mel=mel
        )

    @torch.no_grad()
    def __call__(
        self,
        batch_size: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_inference_steps: int = 50,
        encoding: Optional[torch.FloatTensor] = None, # The conditional tensor
    ) -> Tuple[Optional[List[np.ndarray]], List[np.ndarray]]:

        # 1. Input validation
        if self.unet.config.cross_attention_dim is not None and encoding is None:
            raise ValueError(
                "This UNet is conditional, but no `encoding` was provided."
            )

        # 2. Define the output shape from the UNet config
        shape = (
            batch_size,
            self.unet.config.in_channels,
            self.unet.config.sample_size[0],
            self.unet.config.sample_size[1],
        )
        device = self.device

        # 3. Set up the scheduler and create the initial random noise
        self.scheduler.set_timesteps(num_inference_steps)
        latents = torch.randn(
            shape,
            generator=generator,
            device=device,
            dtype=self.unet.dtype
        )

        # 4. The main denoising loop
        for t in self.progress_bar(self.scheduler.timesteps):
            # Predict the noise residual, passing the conditioning vector
            noise_pred = self.unet(
                sample=latents,
                timestep=t,
                encoder_hidden_states=encoding
            ).sample

            # Use the scheduler to compute the previous (less noisy) sample
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        # 5. Decode the final latents into a spectrogram image
        if self.vqvae is not None:
            decoded_latents = 1 / self.vqvae.config.scaling_factor * latents
            images = self.vqvae.decode(decoded_latents).sample
        else:
            images = latents

        # 6. Post-process the image to convert to PIL
        images = (images / 2 + 0.5).clamp(0, 1)
        images = images.cpu().permute(0, 2, 3, 1).numpy()
        images = (images * 255).round().astype("uint8")
        images = list(
            (Image.fromarray(_[:, :, 0]) for _ in images)
            if images.shape[3] == 1
            else (Image.fromarray(_, mode="RGB").convert("L") for _ in images)
        )

        # 7. Convert the spectrogram image to audio using the mel decoder
        audios = [self.mel.image_to_audio(_) for _ in images]
        
        return images, audios

def hash_image(image):
    image = image.convert("RGB")
    return hashlib.md5(image.tobytes()).hexdigest()

def compute_snr(scheduler, timesteps: torch.LongTensor) -> torch.Tensor:
    """Compute signal-to-noise ratio for the provided timesteps."""
    alphas_cumprod = scheduler.alphas_cumprod.to(device=timesteps.device, dtype=torch.float32)
    alphas = alphas_cumprod[timesteps]
    snr = alphas / (1 - alphas)
    return snr


def scheduled_weight(base_weight: float, start_step: int, ramp_steps: int, current_step: int) -> float:
    if base_weight <= 0.0:
        return 0.0
    if current_step < start_step:
        return 0.0
    if ramp_steps <= 0:
        return float(base_weight)
    progress = (current_step - start_step) / max(ramp_steps, 1)
    progress = min(max(progress, 0.0), 1.0)
    return float(base_weight) * progress


def get_full_repo_name(model_id, organization = None, token = None):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"
    
class PsychoacousticLoss(torch.nn.Module):
    def __init__(self, freq_threshold, clip_threshold):
        super(PsychoacousticLoss, self).__init__()
        self.freq_threshold = freq_threshold
        self.clip_threshold = clip_threshold

    def forward(self, input, target):
        # penalty if the waveform exceeds certain thresholds
        clipping_penalty =  torch.mean(torch.relu(torch.abs(input) - self.clip_threshold))

        # penalty for high frequency content not present in the target
        input_high_freq = torch.mean(input[:, :, self.freq_threshold:], dim=-1)
        target_high_freq = torch.mean(target[:, :, self.freq_threshold:], dim=-1)
        high_freq_penalty = torch.mean(torch.relu(input_high_freq - target_high_freq))

        return clipping_penalty + high_freq_penalty
    
class NISQALoss(torch.nn.Module):
    def __init__(self, alpha=1.0, beta=1.0, gamma=1.0):
        super(NISQALoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.nisqa = NISQA_DIM()
        model_path = 'src/nisqa/nisqa/weights/nisqa.tar'
        checkpoint = torch.load(model_path, map_location='cuda', weights_only=True)
        self.nisqa.load_state_dict(checkpoint['model_state_dict'])

    def forward(self, input):
        # Run inference on the NISQA model
        output = self.nisqa.forward(input.unsqueeze(1), torch.as_tensor([1 for _ in range(input.shape[0])]))
        
        # Extract the NOI, COL, and DISC values
        noi = output[:, 0]
        col = output[:, 1]
        disc = output[:, 2]

        # Compute individual loss terms
        noi_loss = self.alpha * torch.mean(torch.relu(noi))
        col_loss = self.beta * torch.mean(torch.relu(col))
        disc_loss = self.gamma * torch.mean(torch.relu(disc))

        # Combine the losses
        total_loss = noi_loss + col_loss + disc_loss
        
        return total_loss
    
    def mos(self, input):
        # Run inference on the NISQA model
        with torch.no_grad():
            output = self.nisqa.forward(input.unsqueeze(1), torch.as_tensor([1 for _ in range(input.shape[0])]))
        
        # Extract the MOS value
        mos = output[:, 4]
        
        return mos

def main(args: argparse.Namespace):
    output_dir = args.output_dir
    logging_dir = os.path.join(output_dir, args.logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=logging_dir,
        mixed_precision="fp16" if torch.cuda.is_available() else "no"
    )
    
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    # Load dataset
    if args.dataset_name is not None:
        if os.path.exists(args.dataset_name):
            dataset = load_from_disk(
                dataset_path=args.dataset_name,
                storage_options=args.dataset_config_name)["train"]
        else:
            dataset = load_dataset(
                path=args.dataset_name,
                name=args.dataset_config_name,
                token=os.getenv("HF_TOKEN"),
                split="train",
            )
    else:
        raise ValueError("Dataset not found.")
    
    # Deduplicate dataset if requested
    if args.deduplicate:
        seen_hashes = set()
        unique_indices = []
        for i, example in enumerate(dataset):
            h = hash_image(example["image"])
            if h not in seen_hashes:
                seen_hashes.add(h)
                unique_indices.append(i)
        print(f"Deduplicated dataset from {len(dataset)} to {len(unique_indices)} samples.")
        dataset = dataset.select(unique_indices)
    
    # Determine image resolution
    resolution = dataset[0]["spec"].height, dataset[0]["spec"].width

    # Load conditioning encodings (keep CPU-friendly objects only)
    encodings = None
    if args.encodings is not None:
        try:
            with open(args.encodings, "rb") as handle:
                encodings = pickle.load(handle)
        except FileNotFoundError as exc:
            raise FileNotFoundError("Encodings file not found.") from exc

    # Pre-load VQ-VAE before spawning dataloader workers to avoid pickling issues
    vqvae = None
    latent_scaling_factor = 1.0
    latent_resolution = resolution
    convert_to_rgb = False
    if args.vae is not None:
        try:
            vqvae = AutoencoderKL.from_pretrained(args.vae, subfolder="vqvae")
        except EnvironmentError:
            vqvae = clMusDiff.from_pretrained(args.vae, subfolder="vqvae").vqvae
        vqvae: AutoencoderKL = accelerator.prepare(vqvae)
        vqvae.eval()
        vqvae.requires_grad_(False)
        vae_in_channels = getattr(vqvae.config, "in_channels", 1)
        with torch.no_grad():
            dummy_input = torch.zeros(
                (1, vae_in_channels, *resolution),
                device=accelerator.device,
                dtype=torch.float32,
            )
            latent_resolution = vqvae.encode(dummy_input).latent_dist.sample().shape[2:]
        latent_scaling_factor = getattr(vqvae.config, "scaling_factor", 0.18215)
        convert_to_rgb = vae_in_channels == 3

    # Prepare data transforms and augmentation
    augmentations = Compose([
        ToTensor(),
        Normalize([0.5], [0.5]),
    ])

    def transforms(examples):
        if convert_to_rgb:
            spectrograms = [
                augmentations(spec.convert("RGB"))
                for spec in examples["spec"]
            ]
        else:
            spectrograms = [augmentations(spec) for spec in examples["spec"]]

        batch = {"input": spectrograms}
        if encodings is not None:
            batch["encoding"] = [encodings[name] for name in examples["sample_name"]]
        return batch

    dataset.set_transform(transforms)
    
    # Prepare data loader with improved throughput settings
    dataloader_kwargs = dict(
        dataset=dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
        pin_memory=args.pin_memory,
    )
    if args.dataloader_num_workers > 0:
        dataloader_kwargs["persistent_workers"] = args.persistent_workers
        if args.prefetch_factor is not None:
            dataloader_kwargs["prefetch_factor"] = args.prefetch_factor
    train_dataloader = DataLoader(**dataloader_kwargs)

    # Load model from pretrained if available
    if args.pretrained_model is not None:
        pipeline = clMusDiff.from_pretrained(args.pretrained_model)
        mel = pipeline.mel
        model = pipeline.unet
        if hasattr(pipeline, "vqvae"):
            vqvae = pipeline.vqvae
            vqvae: AutoencoderKL = accelerator.prepare(vqvae)
    else: # Initialize new model
        if args.encodings is None: # Unconditional model
            model = UNet2DModel(
                sample_size=resolution if vqvae is None else latent_resolution,
                in_channels=1 if vqvae is None else vqvae.config["latent_channels"],
                out_channels=1 if vqvae is None else vqvae.config["latent_channels"],
                layers_per_block=2,
                block_out_channels=(128, 128, 256, 256, 512, 512),
                down_block_types=(
                    "DownBlock2D",
                    "DownBlock2D",
                    "DownBlock2D",
                    "DownBlock2D",
                    "AttnDownBlock2D",
                    "DownBlock2D",
                ),
                up_block_types=(
                    "UpBlock2D",
                    "AttnUpBlock2D",
                    "UpBlock2D",
                    "UpBlock2D",
                    "UpBlock2D",
                    "UpBlock2D",
                ),
            )
        else: # Conditional model
            model = UNet2DConditionModel(
                sample_size=resolution if vqvae is None else latent_resolution,
                in_channels=1 if vqvae is None else vqvae.config["latent_channels"],
                out_channels=1 if vqvae is None else vqvae.config["latent_channels"],
                layers_per_block=2,
                block_out_channels=(128, 256, 512, 512),
                down_block_types=(
                    "CrossAttnDownBlock2D",
                    "CrossAttnDownBlock2D",
                    "CrossAttnDownBlock2D",
                    "DownBlock2D",
                ),
                up_block_types=(
                    "UpBlock2D",
                    "CrossAttnUpBlock2D",
                    "CrossAttnUpBlock2D",
                    "CrossAttnUpBlock2D",
                ),
                cross_attention_dim=list(encodings.values())[0].shape[-1],
            )

    if args.gradient_checkpointing and hasattr(model, "enable_gradient_checkpointing"):
        model.enable_gradient_checkpointing()

    if args.use_xformers and is_xformers_available():
        try:
            model.enable_xformers_memory_efficient_attention()
        except Exception as exc:
            logger.warning(f"Unable to enable xformers attention: {exc}")
    elif args.use_xformers:
        logger.warning("xFormers is not available; continuing without memory-efficient attention.")

    # Initialize noise scheduler
    if args.scheduler == "ddpm":  # DDPMScheduler
        noise_scheduler = DDPMScheduler(
            num_train_timesteps=args.num_training_timesteps,
            beta_schedule=args.beta_schedule,
            prediction_type=args.prediction_type,
        )
    elif args.scheduler == "ddim":  # DDIMScheduler
        noise_scheduler = DDIMScheduler(
            num_train_timesteps=args.num_training_timesteps,
            beta_schedule=args.beta_schedule,
            prediction_type=args.prediction_type,
        )
    else:
        raise ValueError(f"Unsupported scheduler '{args.scheduler}'.")
    num_inference_steps = args.num_inference_timesteps
    noise_scheduler.set_timesteps(num_inference_steps)

    # Initialize optimizer and learning rate scheduler
    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    num_steps = (len(train_dataloader) * args.num_epochs) // args.gradient_accumulation_steps
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=(args.lr_warmup_percent / 100) * num_steps,
        num_training_steps=num_steps,
        num_cycles=args.lr_warmup_cycles,
    )

    # Prepare model and optimizer for training
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )
    
    # Initialize additional loss term
    if args.perceptual_loss == True:
        perceptual_model = VGGSoundNet(
           pretrained_model='I.pth',
            headless=True, 
        )
        perceptual_model: VGGSoundNet = accelerator.prepare(perceptual_model)
        perceptual_model.eval()
        perceptual_model.requires_grad_(False)
        
    if args.psychoacoustic_loss == True:
        psychoacoustic_loss = PsychoacousticLoss(
            freq_threshold=int(0.85*resolution[0]),
            clip_threshold=0.95,
        )
        psychoacoustic_loss: PsychoacousticLoss = accelerator.prepare(psychoacoustic_loss)
        psychoacoustic_loss.requires_grad_(False)
        
    if args.nisqa_loss == True:
        nisqa_loss = NISQALoss()
        nisqa_loss: NISQALoss = accelerator.prepare(nisqa_loss)
        nisqa_loss.eval()
        nisqa_loss.requires_grad_(False)

    # Initialize EMA model
    ema_model = EMAModel(
        parameters=getattr(model, "module", model).parameters(),
        inv_gamma=args.ema_inv_gamma,
        power=args.ema_power,
        decay=args.ema_decay,
    )

    # Push model to hub if specified
    if args.push_to_hub:
        if args.hub_model_id is None:
            repo_name = get_full_repo_name(Path(output_dir).name,
                                           token=os.getenv("HF_TOKEN"))
        else:
            repo_name = args.hub_model_id
        repo = Repository(output_dir, clone_from=repo_name)

    # Initialize trackers
    if accelerator.is_main_process:
        run = os.path.split(__file__)[-1].split(".")[0]
        accelerator.init_trackers(run)

    # Initialize Mel object
    mel = Mel(
        x_res=resolution[1],
        y_res=resolution[0],
        hop_length=args.hop_length,
        sample_rate=args.sample_rate,
        n_fft=args.n_fft,
    )

    # ================================================================================
    # Train the model
    global_step = 0
    for epoch in range(args.num_epochs): # loop over the dataset for num_epochs
        
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        
        progress_bar = tqdm(
            total=len(train_dataloader),
            disable=not accelerator.is_local_main_process
        )
        progress_bar.set_description(f"Epoch {epoch}")

        # Skip to the start epoch
        if epoch < args.start_epoch:
            for step in range(len(train_dataloader)):
                lr_scheduler.step()
                progress_bar.update(1)
                global_step += 1
            if epoch == args.start_epoch - 1 and args.use_ema:
                ema_model.optimization_step = global_step
            continue

        # Start training
        model.train()
        for step, batch in enumerate(train_dataloader): # loop over samples in the dataset
            
            # print("Input spec:")
            # print(batch["input"][0])
            
            processed_batch = {}
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    processed_batch[key] = value.to(accelerator.device)
                else:
                    processed_batch[key] = value

            if args.encodings is not None and "encoding" in processed_batch:
                encoding_value = processed_batch["encoding"]
                if not isinstance(encoding_value, torch.Tensor):
                    encoding_value = torch.as_tensor(
                        np.asarray(encoding_value),
                        dtype=torch.float32,
                        device=accelerator.device,
                    )
                else:
                    encoding_value = encoding_value.to(accelerator.device, dtype=torch.float32)

                if encoding_value.ndim == 2:
                    encoding_value = encoding_value.unsqueeze(1)
                elif encoding_value.ndim == 1:
                    encoding_value = encoding_value.unsqueeze(0).unsqueeze(0)
                    
                # print("Encoding:")
                # print(encoding_value[0])

                cfg_dropout_mask = None
                if args.cfg_dropout_prob > 0.0:
                    cfg_dropout_mask = torch.rand(
                        encoding_value.shape[0], device=encoding_value.device
                    ) < args.cfg_dropout_prob
                    if cfg_dropout_mask.any():
                        encoding_value = encoding_value.clone()
                        if args.cfg_unconditional_strategy == "zeros":
                            encoding_value[cfg_dropout_mask] = 0
                        elif args.cfg_unconditional_strategy == "input_mean":
                            unconditional_embedding = encoding_value[cfg_dropout_mask].mean(
                                dim=1, keepdim=True
                            )
                            encoding_value[cfg_dropout_mask] = unconditional_embedding

                if cfg_dropout_mask is not None:
                    processed_batch["cfg_dropout_mask"] = cfg_dropout_mask.to(dtype=torch.float32)
                processed_batch["encoding"] = encoding_value

            batch = processed_batch
            gt_specs = batch["input"]

            # Encode spectrograms using the VQ-VAE model
            if vqvae is not None:
                with torch.no_grad():
                    gt_specs_latent = vqvae.encode(gt_specs).latent_dist.sample()
                # print("Latent spec:")
                # print(gt_specs_latent[0])
                gt_specs_latent = gt_specs_latent * latent_scaling_factor
                # print("Scaled latent spec:")
                # print(gt_specs_latent[0])
            else:
                gt_specs_latent = gt_specs

            # Sample noise to add to the clean spectrograms
            noise = torch.randn(gt_specs_latent.shape, device=gt_specs_latent.device)
            if args.use_offset_noise:
                offset_shape = (noise.shape[0], noise.shape[1]) + (1,) * (noise.ndim - 2)
                noise = noise + args.offset_noise_strength * torch.randn(offset_shape, device=noise.device)
            batch_size = gt_specs_latent.shape[0]
            # Sample a random timestep for each image
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (batch_size, ),
                device=gt_specs_latent.device,
            ).long()

            # Add noise to the clean spectrograms according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_specs_latent = noise_scheduler.add_noise(
                original_samples=gt_specs_latent, 
                noise=noise,
                timesteps=timesteps
            )
            # print("Noisy latent spec:")
            # print(noisy_specs_latent[0])

            with accelerator.accumulate(model):
                perceptual_weight = scheduled_weight(
                    args.perceptual_loss_weight,
                    args.perceptual_loss_start_step,
                    args.perceptual_loss_ramp_steps,
                    global_step,
                ) if args.perceptual_loss else 0.0
                psychoacoustic_weight = scheduled_weight(
                    args.psychoacoustic_loss_weight,
                    args.psychoacoustic_loss_start_step,
                    args.psychoacoustic_loss_ramp_steps,
                    global_step,
                ) if args.psychoacoustic_loss else 0.0
                nisqa_weight = scheduled_weight(
                    args.nisqa_loss_weight,
                    args.nisqa_loss_start_step,
                    args.nisqa_loss_ramp_steps,
                    global_step,
                ) if args.nisqa_loss else 0.0

                # Predict the noise residual for the noisy spectrograms
                with accelerator.autocast():
                    if args.encodings is not None:
                        noise_pred = model(
                            sample=noisy_specs_latent,
                            timestep=timesteps,
                            encoder_hidden_states=batch["encoding"],
                        ).sample
                        # print("Noise pred:")
                        # print(noise_pred[0])
                    else:
                        noise_pred = model(noisy_specs_latent, timesteps).sample

                    # ================================================================================
                    # Calculate the loss
                    if noise_scheduler.config.prediction_type == "epsilon":
                        target = noise
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        target = noise_scheduler.get_velocity(gt_specs_latent, noise, timesteps)
                    elif noise_scheduler.config.prediction_type == "sample":
                        target = gt_specs_latent
                    else:
                        raise ValueError(
                            f"Unsupported prediction type {noise_scheduler.config.prediction_type}."
                        )

                    if (
                        args.snr_gamma is not None
                        and args.snr_gamma > 0
                        and noise_scheduler.config.prediction_type in {"epsilon", "v_prediction"}
                    ):
                        snr = compute_snr(noise_scheduler, timesteps)
                        mse_loss = F.mse_loss(noise_pred, target, reduction="none")
                        reduce_dims = list(range(1, mse_loss.ndim))
                        diffusion_loss = mse_loss.mean(dim=reduce_dims)
                        snr = snr.to(device=diffusion_loss.device, dtype=diffusion_loss.dtype)
                        loss_weights = torch.minimum(
                            snr,
                            torch.full_like(snr, args.snr_gamma),
                        ) / snr
                        diffusion_loss = (loss_weights * diffusion_loss).mean()
                    else:
                        diffusion_loss = F.mse_loss(noise_pred, target)
                loss = diffusion_loss

                # Reconstruct predicted clean spectrograms from the model output
                alphas_cumprod = noise_scheduler.alphas_cumprod.to(
                    device=noisy_specs_latent.device,
                    dtype=noisy_specs_latent.dtype,
                )
                alpha_t = alphas_cumprod[timesteps]
                sqrt_alpha_t = alpha_t.sqrt().view(-1, *([1] * (noisy_specs_latent.ndim - 1)))
                sqrt_one_minus_alpha_t = (1 - alpha_t).sqrt().view(
                    -1, *([1] * (noisy_specs_latent.ndim - 1))
                )

                if noise_scheduler.config.prediction_type == "epsilon":
                    pred_original_latent = (
                        noisy_specs_latent - sqrt_one_minus_alpha_t * noise_pred
                    ) / sqrt_alpha_t
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    pred_original_latent = (
                        sqrt_alpha_t * noisy_specs_latent - sqrt_one_minus_alpha_t * noise_pred
                    )
                elif noise_scheduler.config.prediction_type == "sample":
                    pred_original_latent = noise_pred
                else:
                    raise ValueError(
                        f"Unsupported prediction type {noise_scheduler.config.prediction_type}."
                    )
                    
                # print("Predicted original latent spec:")
                # print(pred_original_latent[0])

                # Decode the denoised latent spectrograms
                if vqvae is not None:
                    decoded_latents = 1 / latent_scaling_factor * pred_original_latent
                    # print("Scaled predicted original latent spec:")
                    # print(decoded_latents[0])
                    denoised_specs = vqvae.decode(decoded_latents).sample
                    # print("Denoised spec:")
                    # print(denoised_specs[0])
                else:
                    denoised_specs = pred_original_latent
                denoised_specs = denoised_specs.to(dtype=gt_specs.dtype)
                
                # Add additional loss terms
                if args.perceptual_loss and perceptual_weight > 0:
                    perceptual_loss_value = perceptual_model.perceptual_loss(
                        input=denoised_specs,
                        target=gt_specs,
                    )
                    loss = loss + (perceptual_loss_value * perceptual_weight).detach()
                        
                if args.psychoacoustic_loss and psychoacoustic_weight > 0:
                    psychoacoustic_loss_value = psychoacoustic_loss(
                        input=denoised_specs,
                        target=gt_specs
                    )
                    loss = loss + (psychoacoustic_loss_value * psychoacoustic_weight).detach()

                if args.nisqa_loss and nisqa_weight > 0:
                    nisqa_loss_value = nisqa_loss(
                        input=denoised_specs
                    )
                    loss = loss + (nisqa_loss_value * nisqa_weight).detach()
                
                accelerator.backward(loss)
                # ================================================================================

                # Update the model parameters
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                if args.use_ema:
                    ema_model.step(model.parameters())
                optimizer.zero_grad()

            progress_bar.update(1)
            global_step += 1

            # Log the training progress
            logs = {
                "loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
                "step": global_step,
            }
            if (
                args.nisqa_loss
                and args.mos_log_interval > 0
                and global_step % args.mos_log_interval == 0
            ):
                with torch.no_grad():
                    logs["MOS"] = nisqa_loss.mos(denoised_specs).mean().detach().item()
            logs["diffusion_loss"] = diffusion_loss.detach().item()
            if "cfg_dropout_mask" in batch:
                logs["cfg_dropout_ratio"] = batch["cfg_dropout_mask"].mean().detach().item()
            if args.perceptual_loss and perceptual_weight > 0:
                logs["perceptual_weight"] = perceptual_weight
                logs["perceptual_loss"] = perceptual_loss_value.detach().item()
            if args.psychoacoustic_loss and psychoacoustic_weight > 0:
                logs["psychoacoustic_weight"] = psychoacoustic_weight
                logs["psychoacoustic_loss"] = psychoacoustic_loss_value.detach().item()
            if args.nisqa_loss and nisqa_weight > 0:
                logs["nisqa_weight"] = nisqa_weight
                logs["nisqa_loss"] = nisqa_loss_value.detach().item()
            if args.use_ema:
                logs["ema_decay"] = ema_model.decay
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
        progress_bar.close()

        accelerator.wait_for_everyone()

        # Generate samples for evaluation during training
        if accelerator.is_main_process:
            # Preapre the pipeline for inference
            if ((epoch + 1) % args.save_model_epochs == 0
                    or (epoch + 1) % args.save_samples_epochs == 0
                    or epoch == args.num_epochs - 1):
                unet = accelerator.unwrap_model(model)
                if args.use_ema:
                    ema_model.copy_to(unet.parameters())
                pipeline = clMusDiff(
                    vqvae=vqvae,
                    unet=unet,
                    mel=mel,
                    scheduler=noise_scheduler,
                )
                scheduler_map = {
                    "ddpm": DDPMScheduler,
                    "ddim": DDIMScheduler,
                    "dpmpp_2m": DPMSolverMultistepScheduler,
                    "euler": EulerDiscreteScheduler,
                }
                if args.inference_scheduler in scheduler_map:
                    scheduler_cls = scheduler_map[args.inference_scheduler]
                    pipeline.scheduler = scheduler_cls.from_config(noise_scheduler.config)
                try:
                    pipeline.scheduler.set_timesteps(
                        args.num_inference_timesteps,
                        device=accelerator.device,
                    )
                except TypeError:
                    pipeline.scheduler.set_timesteps(args.num_inference_timesteps)

            # Save the model
            if (epoch + 1) % args.save_model_epochs == 0 or epoch == args.num_epochs - 1:
                pipeline.save_pretrained(output_dir)

                # Push the model to the hub
                if args.push_to_hub:
                    repo.push_to_hub(
                        commit_message=f"Epoch {epoch}",
                        blocking=False,
                        auto_lfs_prune=True,
                    )

            # Generate and save sample spectrograms and audio
            if (epoch + 1) % args.save_samples_epochs == 0:
                generator = torch.Generator(
                    device=gt_specs.device
                ).manual_seed(42)

                if args.encodings is not None:
                    random.seed(42)
                    encoding = torch.stack(
                        [torch.from_numpy(arr) for arr in random.sample(list(encodings.values()), args.num_eval_samples)]
                    ).to(gt_specs.device)
                    encoding = encoding.unsqueeze(1)  # Add sequence dimension: (batch, 1, dim)
                else:
                    encoding = None
                    
                # print("Evaluation encoding:")
                # print(encoding[0] if encoding is not None else None)

                # run pipeline in inference (sample random noise and denoise)
                spectrograms, audios = pipeline(
                    generator=generator,
                    batch_size=args.num_eval_samples,
                    encoding=encoding,
                )

                # print("Generated spec:")
                # print(spectrograms[0])
                # print("Generated audio:")
                # print(audios[0])

                # denormalize the spectrograms and save to tensorboard
                spectrograms = np.array([
                    np.frombuffer(spectrogram.tobytes(), dtype="uint8").reshape(
                        (len(spectrogram.getbands()), spectrogram.height, spectrogram.width)
                    )
                    for spectrogram in spectrograms
                ])
                accelerator.trackers[0].writer.add_images(
                    "test_spectrograms", spectrograms, epoch)
                for _, audio in enumerate(audios):
                    accelerator.trackers[0].writer.add_audio(
                        f"test_audio_{_}",
                        normalize(audio),
                        epoch,
                        sample_rate=pipeline.mel.get_sample_rate(),
                    )
                    
            # Free up some memory
            del spectrograms, audios, generator, encoding, pipeline, unet
               
            
        accelerator.wait_for_everyone()

    accelerator.end_training()
    # ================================================================================


# The above code is a Python script that defines command-line arguments using the `argparse` module.
# It then parses the command-line arguments and assigns them to the `args` variable. Finally, it calls
# the `main` function with the `args` variable as an argument. The purpose of this script is to
# configure and run a training script for a machine learning model.
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Training script for main model.")
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--dataset_name", type=str, default='Woleek/Img2Spec')
    parser.add_argument("--dataset_config_name", type=str, default=None)
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help="A folder containing the training data.",
    )
    parser.add_argument("--output_dir", type=str, default="models/clMusDiffv2")
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--dataloader_num_workers", type=int, default=8)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--pin_memory", type=bool, default=True)
    parser.add_argument("--persistent_workers", type=bool, default=True)
    parser.add_argument("--num_eval_samples", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--save_samples_epochs", type=int, default=1)
    parser.add_argument("--save_model_epochs", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lr_scheduler", type=str, default="cosine")
    parser.add_argument("--lr_warmup_percent", type=int, default=5)
    parser.add_argument("--lr_warmup_cycles", type=int, default=2)
    parser.add_argument("--adam_beta1", type=float, default=0.95)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-6)
    parser.add_argument("--adam_epsilon", type=float, default=1e-08)
    parser.add_argument("--use_ema", type=bool, default=True)
    parser.add_argument("--ema_inv_gamma", type=float, default=1.0)
    parser.add_argument("--ema_power", type=float, default=3 / 4)
    parser.add_argument("--ema_decay", type=float, default=0.9999)
    parser.add_argument("--push_to_hub", type=bool, default=False)
    parser.add_argument("--hub_token", type=str, default=None)
    parser.add_argument("--hub_model_id", type=str, default=None)
    parser.add_argument("--hub_private_repo", type=bool, default=False)
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--hop_length", type=int, default=512)
    parser.add_argument("--sample_rate", type=int, default=22050)
    parser.add_argument("--n_fft", type=int, default=2048)
    parser.add_argument("--pretrained_model", type=str, default=None)
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--num_training_timesteps", type=int, default=1000)
    parser.add_argument("--num_inference_timesteps", type=int, default=50)
    parser.add_argument("--scheduler",
                        type=str,
                        default="ddim",
                        help="ddpm or ddim")
    parser.add_argument("--beta_schedule", type=str, default="scaled_linear")
    parser.add_argument("--prediction_type", type=str, default="epsilon")
    parser.add_argument("--snr_gamma", type=float, default=5.0)
    parser.add_argument(
        "--vae",
        type=str,
        default='teticio/latent-audio-diffusion-256',
        help="pretrained VAE model for latent diffusion",
    )
    parser.add_argument(
        "--encodings",
        type=str,
        default='data/encodings.p',
        help="pickled dictionary mapping audio_file to encoding",
    )
    parser.add_argument(
        "--cfg_dropout_prob",
        type=float,
        default=0.1,
        help="Probability of dropping conditional embeddings for classifier-free guidance training.",
    )
    parser.add_argument(
        "--cfg_unconditional_strategy",
        type=str,
        default="zeros",
        choices=["zeros", "input_mean"],
        help="Replacement strategy when conditioning is dropped during classifier-free guidance training.",
    )
    parser.add_argument("--cfg_scale", type=float, default=2.0, help="Classifier-free guidance scale")
    
    parser.add_argument("--perceptual_loss", type=bool, default=True, help="Use perceptual loss")
    parser.add_argument("--perceptual_loss_weight", type=float, default=0.1, help="Weight for perceptual loss")
    parser.add_argument("--perceptual_loss_start_step", type=int, default=4000, help="Training step to start applying perceptual loss.")
    parser.add_argument("--perceptual_loss_ramp_steps", type=int, default=4000, help="Number of steps to linearly ramp perceptual loss weight.")
    parser.add_argument("--psychoacoustic_loss", type=bool, default=True, help="Use psychoacoustic loss")
    parser.add_argument("--psychoacoustic_loss_weight", type=float, default=0.1, help="Weight for psychoacoustic loss")
    parser.add_argument("--psychoacoustic_loss_start_step", type=int, default=4000, help="Training step to start applying psychoacoustic loss.")
    parser.add_argument("--psychoacoustic_loss_ramp_steps", type=int, default=4000, help="Number of steps to linearly ramp psychoacoustic loss weight.")
    parser.add_argument("--nisqa_loss", type=bool, default=True, help="Use NISQA loss")
    parser.add_argument("--nisqa_loss_weight", type=float, default=0.01, help="Weight for NISQA loss")
    parser.add_argument("--nisqa_loss_start_step", type=int, default=4000, help="Training step to start applying NISQA loss.")
    parser.add_argument("--nisqa_loss_ramp_steps", type=int, default=4000, help="Number of steps to linearly ramp NISQA loss weight.")
    parser.add_argument("--mos_log_interval", type=int, default=100, help="Number of steps between expensive MOS evaluations (set to 1 to log every step).")
    parser.add_argument("--use_offset_noise", type=bool, default=True, help="Add offset noise for improved sample diversity")
    parser.add_argument("--offset_noise_strength", type=float, default=0.1, help="Scaling for offset noise component")
    parser.add_argument("--gradient_checkpointing", type=bool, default=False, help="Enable gradient checkpointing for the UNet")
    parser.add_argument("--use_xformers", type=bool, default=True, help="Use xFormers attention optimizations when available")
    parser.add_argument("--inference_scheduler", type=str, default="dpmpp_2m", help="Scheduler used for evaluation sampling")
    parser.add_argument("--deduplicate", type=bool, default=False, help="Deduplicate dataset by selecting unique images based on hash.")

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError(
            "You must specify either a dataset name from the hub or a train data directory."
        )

    main(args)
