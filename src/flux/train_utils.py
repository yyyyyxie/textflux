import torch
from einops import rearrange

def get_im_mask(img_cond, mask):
    bs, _, _, _ = img_cond.shape
    mask = rearrange(mask, "1 h w -> 1 1 h w")  # 1 1 1632 1232
    img_cond = img_cond.float()
    mask = mask.float()
    img_cond = img_cond * (1 - mask)

    return img_cond

def prepare_fill_with_mask(
        image_processor,
        mask_processor,
        vae,
        vae_scale_factor,
        image,
        mask,
        width,
        height,
        batch_size,
        num_images_per_prompt,
        device,
        dtype,
    ):
    """
    Prepares image and mask for fill operation with proper rearrangement.
    Focuses only on image and mask processing.
    """
    # Determine effective batch size
    effective_batch_size = batch_size * num_images_per_prompt

    mask = rearrange(mask, "1 h w -> 1 1 h w")  # 1 1 1632 1232
    
    # Prepare image
    if isinstance(image, torch.Tensor):
        pass
    else:
        image = image_processor.preprocess(image, height=height, width=width)

    image_batch_size = image.shape[0]
    repeat_by = effective_batch_size if image_batch_size == 1 else num_images_per_prompt
    image = image.repeat_interleave(repeat_by, dim=0)
    image = image.to(device=device, dtype=dtype)

    # Prepare mask with specific processing
    if isinstance(mask, torch.Tensor):
        pass
    else:
        mask = mask_processor.preprocess(mask, height=height, width=width)

    mask = mask.repeat_interleave(repeat_by, dim=0)
    mask = mask.to(device=device, dtype=dtype)

    # Apply mask to image
    masked_image = image.clone()
    masked_image = masked_image * (1 - mask)

    # Encode to latents
    image_latents = vae.encode(masked_image.to(vae.dtype)).latent_dist.sample()
    image_latents = (
        image_latents - vae.config.shift_factor
    ) * vae.config.scaling_factor
    image_latents = image_latents.to(dtype)

    # Process mask following the example's specific rearrangement
    mask = mask[:, 0, :, :] if mask.shape[1] > 1 else mask[:, 0, :, :]
    mask = mask.to(torch.bfloat16)
    
    # First rearrangement: 8x8 patches
    mask = rearrange(
        mask,
        "b (h ph) (w pw) -> b (ph pw) h w",
        ph=8,
        pw=8,
    )
    
    # Second rearrangement: 2x2 patches
    mask = rearrange(
        mask, 
        "b c (h ph) (w pw) -> b (h w) (c ph pw)", 
        ph=2, 
        pw=2
    )

    # Rearrange image latents similarly
    image_latents = rearrange(
        image_latents,
        "b c (h ph) (w pw) -> b (h w) (c ph pw)",
        ph=2,
        pw=2
    )

    # Combine image and mask
    image_cond = torch.cat([image_latents, mask], dim=-1)

    return image_cond, height, width


# Copied from diffusers.pipelines.flux.pipeline_flux._pack_latents
def pack_latents(latents, batch_size, num_channels_latents, height, width):
    latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)

    return latents


def prepare_latent_image_ids(batch_size, height, width, device, dtype):
    latent_image_ids = torch.zeros(height // 2, width // 2, 3)
    latent_image_ids[..., 1] = (
        latent_image_ids[..., 1] + torch.arange(height // 2)[:, None]
    )
    latent_image_ids[..., 2] = (
        latent_image_ids[..., 2] + torch.arange(width // 2)[None, :]
    )

    (
        latent_image_id_height,
        latent_image_id_width,
        latent_image_id_channels,
    ) = latent_image_ids.shape

    latent_image_ids = latent_image_ids.reshape(
        latent_image_id_height * latent_image_id_width,
        latent_image_id_channels,
    )

    return latent_image_ids.to(device=device, dtype=dtype)


def prepare_latents(
    vae_scale_factor,
    batch_size,
    height,
    width,
    dtype,
    device,
):
    height = 2 * (int(height) // (vae_scale_factor * 2))
    width = 2 * (int(width) // (vae_scale_factor * 2))


    latent_image_ids = prepare_latent_image_ids(
        batch_size, height, width, device, dtype
    )
    return latent_image_ids

def decode_packed_image(
    packed_control_image,
    vae,
    vae_scale_factor,
    height,
    width,
    batch_size,
    num_images_per_prompt,
    device,
    dtype,
):
    # Unpack latents
    control_image = unpack_latents(
        packed_control_image,
        batch_size * num_images_per_prompt,
        5,  # 4 channels for image_latents + 1 for mask
        height // vae_scale_factor * 2,
        width // vae_scale_factor * 2,
    )

    # Split control_image into image_latents and mask
    image_latents, mask = torch.split(control_image, [4, 1], dim=1)

    # Decode latents
    image_latents = image_latents / vae.config.scaling_factor + vae.config.shift_factor
    image = vae.decode(image_latents.to(vae.dtype)).sample

    # Interpolate mask back to original size
    mask = torch.nn.functional.interpolate(mask, size=(height, width))
    mask = 1 - mask  # Invert mask

    # Apply mask to image
    masked_image = image.clone()
    masked_image[(mask > 0.5).repeat(1, 3, 1, 1)] = -1

    return image, masked_image, mask

# Helper function to unpack latents
def unpack_latents(packed_latents, batch_size, num_channels, height, width):
    unpacked = packed_latents.reshape(batch_size, height // 2, width // 2, num_channels, 2, 2)
    unpacked = unpacked.permute(0, 3, 1, 4, 2, 5)
    unpacked = unpacked.reshape(batch_size, num_channels, height, width)
    return unpacked


def get_image_proj(
    transformer,
    image_prompt: torch.Tensor,
    device,
):
    if transformer.auto_processor is not None and transformer.image_encoder is not None and transformer.garment_adapter_improj is not None:
        # encode image-prompt embeds
        # transformer.image_encoder.to(device=device, dtype=torch.float32)
        # print("image_prompt.dtype", image_prompt.dtype)
        image_prompt = transformer.clip_image_processor(
            images=image_prompt,
            return_tensors="pt"
        ).pixel_values
        
        image_prompt = image_prompt.to(device)
        image_prompt_embeds = transformer.image_encoder(
            image_prompt
        ).image_embeds.to(
            device=device, dtype=torch.bfloat16,
        )
        
        # encode image
        # print("image_prompt_embeds.shape", image_prompt_embeds.shape)
        image_proj = transformer.garment_adapter_improj(image_prompt_embeds)
        
        return image_proj
    else:
        print("No image projector found")
        return None
    
    
def encode_images_to_latents(vae, pixel_values, weight_dtype, height, width, image_processor=None):
    if image_processor is not None:
        pixel_values = image_processor.preprocess(pixel_values, height=height, width=width).to(dtype=vae.dtype, device=vae.device)
    
    pixel_values = pixel_values.to(dtype=weight_dtype, device=vae.device)
    model_input = vae.encode(pixel_values).latent_dist.sample()
    model_input = (model_input - vae.config.shift_factor) * vae.config.scaling_factor
    model_input = model_input.to(dtype=weight_dtype)
    
    return model_input


@staticmethod
def _unpack_latents(latents, height, width, vae_scale_factor):
    batch_size, num_patches, channels = latents.shape

    height = height // vae_scale_factor
    width = width // vae_scale_factor

    latents = latents.view(batch_size, height, width, channels // 4, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)

    latents = latents.reshape(
        batch_size, channels // (2 * 2), height * 2, width * 2
    )

    return latents