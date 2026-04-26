import argparse
import copy
import time
import random
import sys
from pathlib import Path

import imageio.v3 as iio
import toml
import torch
import torchvision
from diffusers import FlowMatchEulerDiscreteScheduler

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# Keep behavior aligned with train.py.
sys.modules["comfy_kitchen"] = None

# Import repo utils before models.base adds ComfyUI to the front of sys.path.
from utils import common
from models.base import PreprocessMediaFile
from models.wan.wan import WanPipeline
from models.wan import configs as wan_configs


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run minimal Wan i2v inference from a random sample in a diffusion-pipe dataset."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "examples" / "libero_wan_fun_1p3b_i2v_65_fft.toml",
        help="Training config used for the fine-tuned model.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Specific saved model directory containing model.safetensors. Defaults to the latest saved model under the latest run.",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=None,
        help="Dataset index to use. If omitted, a random sample is chosen.",
    )
    parser.add_argument(
        "--sample-path",
        type=Path,
        default=None,
        help="Explicit video path to use instead of sampling from the dataset.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Number of sampling steps. Defaults to official-like Wan i2v settings.",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=None,
        help="Classifier-free guidance scale. Defaults to official-like Wan i2v settings.",
    )
    parser.add_argument(
        "--sample-shift",
        type=float,
        default=None,
        help="Flow-match scheduler shift. Defaults to official-like Wan i2v settings.",
    )
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default=None,
        help="Negative prompt. Defaults to Wan's built-in i2v negative prompt.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "inference_outputs",
        help="Directory where outputs will be written.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Torch device to use, usually cuda or cuda:0.",
    )
    return parser.parse_args()


def find_latest_dir(parent: Path):
    candidates = [p for p in parent.iterdir() if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No run directories found under {parent}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def find_latest_model_dir(run_dir: Path):
    candidates = []
    for child in run_dir.iterdir():
        if child.is_dir() and (child / "model.safetensors").exists():
            candidates.append(child)
    if not candidates:
        raise FileNotFoundError(f"No saved model directories with model.safetensors found under {run_dir}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def load_train_and_dataset_config(config_path: Path):
    with open(config_path) as f:
        train_config = toml.load(f)

    model_config = train_config["model"]
    if isinstance(model_config.get("dtype"), str):
        model_config["dtype"] = common.DTYPE_MAP[model_config["dtype"]]
    if isinstance(model_config.get("transformer_dtype"), str):
        model_config["transformer_dtype"] = common.DTYPE_MAP[model_config["transformer_dtype"]]

    dataset_path = Path(train_config["dataset"])
    with open(dataset_path) as f:
        dataset_config = toml.load(f)
    return train_config, dataset_config


def resolve_checkpoint_dir(train_config, checkpoint_dir: Path | None):
    if checkpoint_dir is not None:
        checkpoint_dir = checkpoint_dir.expanduser().resolve()
        if not (checkpoint_dir / "model.safetensors").exists():
            raise FileNotFoundError(f"Could not find model.safetensors in {checkpoint_dir}")
        return checkpoint_dir

    output_dir = Path(train_config["output_dir"])
    run_dir = find_latest_dir(output_dir)
    return find_latest_model_dir(run_dir)


def collect_dataset_videos(dataset_config):
    videos = []
    for directory in dataset_config["directory"]:
        dataset_root = Path(directory["path"])
        videos.extend(sorted(dataset_root.glob("*.mp4")))
    if not videos:
        raise FileNotFoundError("No .mp4 files found in dataset directories from the dataset config.")
    return videos


def choose_sample(args, dataset_config):
    if args.sample_path is not None:
        sample_path = args.sample_path.expanduser().resolve()
        if not sample_path.exists():
            raise FileNotFoundError(f"Sample video path does not exist: {sample_path}")
        return sample_path

    videos = collect_dataset_videos(dataset_config)
    if args.sample_index is not None:
        if args.sample_index < 0 or args.sample_index >= len(videos):
            raise IndexError(f"sample-index {args.sample_index} out of range for dataset of size {len(videos)}")
        return videos[args.sample_index]

    rng = random.Random(args.seed)
    return rng.choice(videos)


def load_caption(video_path: Path):
    caption_path = video_path.with_suffix(".txt")
    if not caption_path.exists():
        raise FileNotFoundError(f"Caption file not found for {video_path}")
    return caption_path.read_text().strip()


def load_clip_from_video(preprocessor, video_path: Path, size_bucket):
    items = preprocessor((None, video_path), None, size_bucket=size_bucket)
    if not items:
        raise RuntimeError(f"No clips were extracted from {video_path}")
    return items[0][0]


def save_video_tensor(video, output_path: Path, fps: int):
    assert video.ndim == 5 and video.shape[0] == 1, video.shape
    video = video.squeeze(0).detach().cpu().to(torch.float32)
    video = ((video + 1.0) / 2.0).clamp(0, 1)
    video = (video * 255).to(torch.uint8)
    video = video.permute(1, 2, 3, 0).contiguous().numpy()
    iio.imwrite(output_path, video, fps=fps)


def save_first_frame(video, output_path: Path):
    assert video.ndim == 5 and video.shape[0] == 1, video.shape
    frame = video.squeeze(0)[:, 0].detach().cpu().to(torch.float32)
    frame = ((frame + 1.0) / 2.0).clamp(0, 1)
    pil_image = torchvision.transforms.functional.to_pil_image(frame)
    pil_image.save(output_path)


def build_inference_pipeline(train_config, checkpoint_dir: Path, device: torch.device):
    inference_config = copy.deepcopy(train_config)
    model_config = inference_config["model"]
    model_config["transformer_path"] = str(checkpoint_dir / "model.safetensors")
    model_config["cache_text_embeddings"] = True

    model_dtype = model_config["dtype"]
    common.AUTOCAST_DTYPE = model_dtype

    pipeline = WanPipeline(inference_config)
    pipeline.load_diffusion_model()

    pipeline.transformer.eval().to(device)
    pipeline.text_encoder.model.eval().to(device)
    pipeline.vae.model.eval().to(device)
    if hasattr(pipeline, "clip"):
        pipeline.clip.model.eval().to(device)

    return pipeline


def encode_texts(pipeline, captions: list[str], device: torch.device):
    ids, mask = pipeline.text_encoder.tokenizer(captions, return_mask=True, add_special_tokens=True)
    ids = ids.to(device)
    mask = mask.to(device)
    seq_lens = mask.gt(0).sum(dim=1).long()
    p = next(pipeline.text_encoder.model.parameters())
    with torch.autocast(device_type=device.type, dtype=p.dtype):
        text_embeddings = pipeline.text_encoder.model(ids, mask)
    return text_embeddings, seq_lens


def encode_conditioning(pipeline, clip_tensor: torch.Tensor, device: torch.device):
    p = next(pipeline.vae.model.parameters())
    clip_tensor = clip_tensor.unsqueeze(0).to(device, p.dtype)
    cond_video = torch.zeros_like(clip_tensor)
    cond_video[:, :, 0:1] = clip_tensor[:, :, 0:1]

    with torch.inference_mode():
        y = pipeline.vae.model.encode(cond_video, pipeline.vae.scale)
        clip_context = pipeline.clip.visual(clip_tensor[:, :, 0:1].to(device, p.dtype))
    return y, clip_context, clip_tensor


def transformer_forward(layers, x_t, y, t_ms, text_embeddings, seq_lens, clip_context):
    hidden = layers[0]((x_t, y, t_ms, text_embeddings, seq_lens, clip_context))
    for layer in layers[1:-1]:
        hidden = layer(hidden)
    return layers[-1](hidden)


def get_official_like_i2v_settings(pipeline, args):
    if pipeline.model_type not in ("i2v", "i2v_v2", "flf2v"):
        raise RuntimeError(f"This script expects an i2v-style model, got {pipeline.model_type}")

    # Wan2.2 exposes explicit sampling defaults in configs.py. Wan2.1 1.3B i2v is a local hack path,
    # so we fall back to the closest local Wan i2v defaults we have.
    if pipeline.model_type == "i2v_v2":
        official_cfg = wan_configs.i2v_A14B
        default_steps = official_cfg.sample_steps
        default_shift = official_cfg.sample_shift
        default_guidance = official_cfg.sample_guide_scale
        if isinstance(default_guidance, tuple):
            default_guidance = default_guidance[-1]
        default_negative_prompt = official_cfg.sample_neg_prompt
    else:
        default_steps = 40
        default_shift = 5.0
        default_guidance = 3.5
        default_negative_prompt = wan_configs.i2v_14B.sample_neg_prompt

    return {
        "steps": args.steps if args.steps is not None else default_steps,
        "sample_shift": args.sample_shift if args.sample_shift is not None else default_shift,
        "guidance_scale": args.guidance_scale if args.guidance_scale is not None else default_guidance,
        "negative_prompt": args.negative_prompt if args.negative_prompt is not None else default_negative_prompt,
    }


def sample_video(
    pipeline,
    text_embeddings,
    seq_lens,
    y,
    clip_context,
    steps: int,
    sample_shift: float,
    guidance_scale: float,
    seed: int,
    device: torch.device,
):
    if steps < 1:
        raise ValueError("steps must be >= 1")

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    scheduler = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=wan_configs.wan_shared_cfg.num_train_timesteps,
        shift=sample_shift,
        use_dynamic_shifting=False,
    )
    scheduler.set_timesteps(steps, device=device)

    layers = pipeline.to_layers()
    x_t = torch.randn(y.shape, device=device, dtype=torch.float32, generator=generator)
    y_cfg = torch.cat([y, y], dim=0)
    clip_cfg = torch.cat([clip_context, clip_context], dim=0)

    with torch.inference_mode():
        for timestep in scheduler.timesteps:
            model_input = torch.cat([x_t, x_t], dim=0)
            t_ms = timestep.expand(model_input.shape[0]).to(device=device, dtype=torch.float32)
            model_output = transformer_forward(layers, model_input, y_cfg, t_ms, text_embeddings, seq_lens, clip_cfg).float()
            noise_uncond, noise_cond = model_output.chunk(2, dim=0)
            guided = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
            x_t = scheduler.step(guided, timestep, x_t, generator=generator, return_dict=False)[0]

    return x_t


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this script.")

    train_config, dataset_config = load_train_and_dataset_config(args.config)
    checkpoint_dir = resolve_checkpoint_dir(train_config, args.checkpoint_dir)
    sample_path = choose_sample(args, dataset_config)
    caption = load_caption(sample_path)

    size_bucket = tuple(dataset_config["size_buckets"][0])
    device = torch.device(args.device)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    total_start = time.perf_counter()
    pipeline = build_inference_pipeline(train_config, checkpoint_dir, device)

    preprocessor = pipeline.get_preprocess_media_file_fn()
    clip_tensor = load_clip_from_video(preprocessor, sample_path, size_bucket=size_bucket)

    inference_start = time.perf_counter()
    sampling_settings = get_official_like_i2v_settings(pipeline, args)
    text_embeddings, seq_lens = encode_texts(
        pipeline,
        [sampling_settings["negative_prompt"], caption],
        device,
    )
    y, clip_context, processed_clip = encode_conditioning(pipeline, clip_tensor, device)
    latents = sample_video(
        pipeline,
        text_embeddings=text_embeddings,
        seq_lens=seq_lens,
        y=y,
        clip_context=clip_context,
        steps=sampling_settings["steps"],
        sample_shift=sampling_settings["sample_shift"],
        guidance_scale=sampling_settings["guidance_scale"],
        seed=args.seed,
        device=device,
    )

    vae_dtype = next(pipeline.vae.model.parameters()).dtype
    with torch.inference_mode():
        decoded = pipeline.vae.model.decode(latents.to(vae_dtype), pipeline.vae.scale)
    inference_time_sec = time.perf_counter() - inference_start
    total_time_sec = time.perf_counter() - total_start

    run_name = checkpoint_dir.parent.name
    model_name = checkpoint_dir.name
    output_dir = args.output_dir / f"{run_name}_{model_name}_{sample_path.stem}"
    output_dir.mkdir(parents=True, exist_ok=True)

    save_video_tensor(decoded, output_dir / "generated.mp4", fps=pipeline.framerate)
    save_video_tensor(processed_clip, output_dir / "source_clip.mp4", fps=pipeline.framerate)
    save_first_frame(processed_clip, output_dir / "initial_frame.png")
    (output_dir / "prompt.txt").write_text(caption + "\n")
    (output_dir / "negative_prompt.txt").write_text(sampling_settings["negative_prompt"] + "\n")
    (output_dir / "sample_path.txt").write_text(str(sample_path) + "\n")

    print(f"Using checkpoint dir: {checkpoint_dir}")
    print(f"Using sample: {sample_path}")
    print(f"Prompt: {caption}")
    print(f"Negative prompt: {sampling_settings['negative_prompt']}")
    print("Sampler: FlowMatchEulerDiscreteScheduler")
    print(f"Sampling steps: {sampling_settings['steps']}")
    print(f"Guidance scale: {sampling_settings['guidance_scale']}")
    print(f"Sample shift: {sampling_settings['sample_shift']}")
    print(f"Inference time: {inference_time_sec:.3f} s")
    print(f"Total script time: {total_time_sec:.3f} s")
    print(f"Saved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
