# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path
import os
import time
import torch
import subprocess
from PIL import Image
from typing import List
from diffusers import StableDiffusionXLPipeline

# Change model name here:
MODEL_NAME = "CinematicRedmond.safetensors"
DEFAULT_INFERENCE_STEPS = 30
DEFAULT_GUIDANCE_SCALE = 6
MODEL_CACHE = "checkpoints/"
MODEL_URL = "https://weights.replicate.delivery/default//artificialguybr/CinematicRedmond.tar"

def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-x", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)

class Predictor(BasePredictor):
    def create_pipeline(
        self,
        pipeline_class,
        safety_checker: bool = True,
    ):
        kwargs = {
            "cache_dir": MODEL_CACHE,
            "torch_dtype" : torch.float16,
        }
        if not safety_checker:
            kwargs["safety_checker"] = None

        pipe = pipeline_class.from_single_file(MODEL_CACHE+MODEL_NAME, **kwargs)
        pipe.to('cuda')
        pipe.enable_xformers_memory_efficient_attention()
        return pipe

    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        # Download with Pget
        if not os.path.exists(MODEL_CACHE):
            download_weights(MODEL_URL, MODEL_CACHE)
        self.txt2img_pipe = self.create_pipeline(StableDiffusionXLPipeline)
        self.txt2img_pipe_unsafe = self.create_pipeline(
            StableDiffusionXLPipeline, safety_checker=False
        )
        # warm up pipes
        # self.txt2img_pipe(prompt="warmup")
        # self.txt2img_pipe_unsafe(prompt="warmup")

    def get_dimensions(self, image):
        original_width, original_height = image.size
        print(
            f"Original dimensions: Width: {original_width}, Height: {original_height}"
        )
        resized_width, resized_height = self.get_resized_dimensions(
            original_width, original_height
        )
        print(
            f"Dimensions to resize to: Width: {resized_width}, Height: {resized_height}"
        )
        return resized_width, resized_height

    def get_allowed_dimensions(self, base=512, max_dim=1024):
        """
        Function to generate allowed dimensions optimized around a base up to a max
        """
        allowed_dimensions = []
        for i in range(base, max_dim + 1, 64):
            for j in range(base, max_dim + 1, 64):
                allowed_dimensions.append((i, j))
        return allowed_dimensions

    def get_resized_dimensions(self, width, height):
        allowed_dimensions = self.get_allowed_dimensions()
        aspect_ratio = width / height
        print(f"Aspect Ratio: {aspect_ratio:.2f}")
        # Find the closest allowed dimensions that maintain the aspect ratio
        # and are closest to the optimum dimension of 768
        optimum_dimension = 768
        closest_dimensions = min(
            allowed_dimensions,
            key=lambda dim: abs(dim[0] / dim[1] - aspect_ratio)
            + abs(dim[0] - optimum_dimension),
        )
        return closest_dimensions
    
    @torch.inference_mode()
    def predict(
        self,
        prompt: str = Input(
            description="Input prompt",
            default="Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
        ),
        negative_prompt: str = Input(
            description="Input Negative Prompt",
            default="text, watermark, blur, deformed, noised",
        ),
        width: int = Input(
            description="Width of output image. Lower if out of memory",
            default=1656,
        ),
        height: int = Input(
            description="Height of output image. Lower if out of memory",
            default=744,
        ),
        num_images: int = Input(
            description="Number of images per prompt",
            ge=1,
            le=8,
            default=1,
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps",
            ge=1,
            le=100,
            default=DEFAULT_INFERENCE_STEPS,
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance", ge=1, le=20, default=DEFAULT_GUIDANCE_SCALE
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
        disable_safety_checker: bool = Input(
            description="Disable safety checker for generated images. This feature is only available through the API",
            default=False,
        ),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        prediction_start = time.time()

        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")

        print(f"Using seed: {seed}")
        os.environ['PYTHONHASHSEED'] = str(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

        kwargs = {}
        mode = "txt2img"
        print(f"{mode} mode")
        pipe = getattr(
            self,
            f"{mode}_pipe" if not disable_safety_checker else f"{mode}_pipe_unsafe",
        )

        common_args = {
            "width": width,
            "height": height,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "guidance_scale": guidance_scale,
            "num_images_per_prompt": num_images,
            "num_inference_steps": num_inference_steps,
            "output_type": "pil",
        }

        start = time.time()
        result = pipe(
            **common_args,
            **kwargs,
            generator=torch.Generator("cuda").manual_seed(seed),
        ).images
        print(f"Inference took: {time.time() - start:.2f}s")

        output_paths = []
        for i, sample in enumerate(result):
            output_path = f"/tmp/out-{i}.jpg"
            sample.save(output_path)
            output_paths.append(Path(output_path))

        print(f"Prediction took: {time.time() - prediction_start:.2f}s")
        return output_paths
