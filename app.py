import os
from portable_env import APP_ROOT, configure_portable_environment

PORTABLE_PATHS = configure_portable_environment()

import gradio as gr
from datetime import datetime
import shutil
import cv2
import threading
from typing import *
import torch
import numpy as np
from PIL import Image
import base64
import io
from trellis2.modules.sparse import SparseTensor
from trellis2.pipelines import Trellis2ImageTo3DPipeline
from trellis2.renderers import EnvMap
from trellis2.utils import render_utils
import o_voxel


MAX_SEED = np.iinfo(np.int32).max
TMP_DIR = str(PORTABLE_PATHS['sessions'])
MODES = [
    {"name": "Normal", "icon": APP_ROOT / "assets/app/normal.png", "render_key": "normal"},
    {"name": "Clay render", "icon": APP_ROOT / "assets/app/clay.png", "render_key": "clay"},
    {"name": "Base color", "icon": APP_ROOT / "assets/app/basecolor.png", "render_key": "base_color"},
    {"name": "HDRI forest", "icon": APP_ROOT / "assets/app/hdri_forest.png", "render_key": "shaded_forest"},
    {"name": "HDRI sunset", "icon": APP_ROOT / "assets/app/hdri_sunset.png", "render_key": "shaded_sunset"},
    {"name": "HDRI courtyard", "icon": APP_ROOT / "assets/app/hdri_courtyard.png", "render_key": "shaded_courtyard"},
]
STEPS = 8
DEFAULT_MODE = 3
DEFAULT_STEP = 3
pipeline = None
envmap = None
_runtime_lock = threading.Lock()
_ui_assets_lock = threading.Lock()


css = """
/* Overwrite Gradio Default Style */
.stepper-wrapper {
    padding: 0;
}

.stepper-container {
    padding: 0;
    align-items: center;
}

.step-button {
    flex-direction: row;
}

.step-connector {
    transform: none;
}

.step-number {
    width: 16px;
    height: 16px;
}

.step-label {
    position: relative;
    bottom: 0;
}

.wrap.center.full {
    inset: 0;
    height: 100%;
}

.wrap.center.full.translucent {
    background: var(--block-background-fill);
}

.meta-text-center {
    display: block !important;
    position: absolute !important;
    top: unset !important;
    bottom: 0 !important;
    right: 0 !important;
    transform: unset !important;
}

/* Previewer */
.previewer-container {
    position: relative;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    width: 100%;
    height: 722px;
    margin: 0 auto;
    padding: 20px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
}

.previewer-container .tips-icon {
    position: absolute;
    right: 10px;
    top: 10px;
    z-index: 10;
    border-radius: 10px;
    color: #fff;
    background-color: var(--color-accent);
    padding: 3px 6px;
    user-select: none;
}

.previewer-container .tips-text {
    position: absolute;
    right: 10px;
    top: 50px;
    color: #fff;
    background-color: var(--color-accent);
    border-radius: 10px;
    padding: 6px;
    text-align: left;
    max-width: 300px;
    z-index: 10;
    transition: all 0.3s;
    opacity: 0%;
    user-select: none;
}

.previewer-container .tips-text p {
    font-size: 14px;
    line-height: 1.2;
}

.tips-icon:hover + .tips-text { 
    display: block;
    opacity: 100%;
}

/* Row 1: Display Modes */
.previewer-container .mode-row {
    width: 100%;
    display: flex;
    gap: 8px;
    justify-content: center;
    margin-bottom: 20px;
    flex-wrap: wrap;
}
.previewer-container .mode-btn {
    width: 24px;
    height: 24px;
    border-radius: 50%;
    cursor: pointer;
    opacity: 0.5;
    transition: all 0.2s;
    border: 2px solid #ddd;
    object-fit: cover;
}
.previewer-container .mode-btn:hover { opacity: 0.9; transform: scale(1.1); }
.previewer-container .mode-btn.active {
    opacity: 1;
    border-color: var(--color-accent);
    transform: scale(1.1);
}

/* Row 2: Display Image */
.previewer-container .display-row {
    margin-bottom: 20px;
    min-height: 400px;
    width: 100%;
    flex-grow: 1;
    display: flex;
    justify-content: center;
    align-items: center;
}
.previewer-container .previewer-main-image {
    max-width: 100%;
    max-height: 100%;
    flex-grow: 1;
    object-fit: contain;
    display: none;
}
.previewer-container .previewer-main-image.visible {
    display: block;
}

/* Row 3: Custom HTML Slider */
.previewer-container .slider-row {
    width: 100%;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 10px;
    padding: 0 10px;
}

.previewer-container input[type=range] {
    -webkit-appearance: none;
    width: 100%;
    max-width: 400px;
    background: transparent;
}
.previewer-container input[type=range]::-webkit-slider-runnable-track {
    width: 100%;
    height: 8px;
    cursor: pointer;
    background: #ddd;
    border-radius: 5px;
}
.previewer-container input[type=range]::-webkit-slider-thumb {
    height: 20px;
    width: 20px;
    border-radius: 50%;
    background: var(--color-accent);
    cursor: pointer;
    -webkit-appearance: none;
    margin-top: -6px;
    box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    transition: transform 0.1s;
}
.previewer-container input[type=range]::-webkit-slider-thumb:hover {
    transform: scale(1.2);
}

/* Overwrite Previewer Block Style */
.gradio-container .padded:has(.previewer-container) {
    padding: 0 !important;
}

.gradio-container:has(.previewer-container) [data-testid="block-label"] {
    position: absolute;
    top: 0;
    left: 0;
}
"""


head = """
<script>
    function refreshView(mode, step) {
        // 1. Find current mode and step
        const allImgs = document.querySelectorAll('.previewer-main-image');
        for (let i = 0; i < allImgs.length; i++) {
            const img = allImgs[i];
            if (img.classList.contains('visible')) {
                const id = img.id;
                const [_, m, s] = id.split('-');
                if (mode === -1) mode = parseInt(m.slice(1));
                if (step === -1) step = parseInt(s.slice(1));
                break;
            }
        }
        
        // 2. Hide ALL images
        // We select all elements with class 'previewer-main-image'
        allImgs.forEach(img => img.classList.remove('visible'));

        // 3. Construct the specific ID for the current state
        // Format: view-m{mode}-s{step}
        const targetId = 'view-m' + mode + '-s' + step;
        const targetImg = document.getElementById(targetId);

        // 4. Show ONLY the target
        if (targetImg) {
            targetImg.classList.add('visible');
        }

        // 5. Update Button Highlights
        const allBtns = document.querySelectorAll('.mode-btn');
        allBtns.forEach((btn, idx) => {
            if (idx === mode) btn.classList.add('active');
            else btn.classList.remove('active');
        });
    }
    
    // --- Action: Switch Mode ---
    function selectMode(mode) {
        refreshView(mode, -1);
    }
    
    // --- Action: Slider Change ---
    function onSliderChange(val) {
        refreshView(-1, parseInt(val));
    }
</script>
"""


empty_html = f"""
<div class="previewer-container">
    <svg style=" opacity: .5; height: var(--size-5); color: var(--body-text-color);"
    xmlns="http://www.w3.org/2000/svg" width="100%" height="100%" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" class="feather feather-image"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect><circle cx="8.5" cy="8.5" r="1.5"></circle><polyline points="21 15 16 10 5 21"></polyline></svg>
</div>
"""


def image_to_base64(image):
    buffered = io.BytesIO()
    image = image.convert("RGB")
    image.save(buffered, format="jpeg", quality=85)
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/jpeg;base64,{img_str}"


def initialize_ui_assets() -> None:
    """Load small local UI assets without initializing the 4B model."""
    if all('icon_base64' in mode for mode in MODES):
        return
    with _ui_assets_lock:
        for mode in MODES:
            if 'icon_base64' not in mode:
                with Image.open(mode['icon']) as icon:
                    mode['icon_base64'] = image_to_base64(icon)


def initialize_runtime() -> None:
    """Load TRELLIS.2 once, on demand, in a thread-safe manner."""
    global pipeline, envmap
    if pipeline is not None:
        return
    with _runtime_lock:
        if pipeline is not None:
            return
        if not torch.cuda.is_available():
            raise gr.Error("The NVIDIA GPU is unavailable. Check the driver and rerun portable setup.")
        gpu = torch.cuda.get_device_properties(0)
        if gpu.total_memory < 23 * 1024**3:
            raise gr.Error(f"TRELLIS.2 needs about 24 GB of VRAM; {gpu.name} has {gpu.total_memory / 1024**3:.1f} GB.")

        initialize_ui_assets()
        loaded_pipeline = Trellis2ImageTo3DPipeline.from_pretrained('microsoft/TRELLIS.2-4B')
        loaded_pipeline.cuda()
        loaded_envmap = {
            'forest': EnvMap(torch.tensor(
                cv2.cvtColor(cv2.imread(str(APP_ROOT / 'assets/hdri/forest.exr'), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB),
                dtype=torch.float32, device='cuda'
            )),
            'sunset': EnvMap(torch.tensor(
                cv2.cvtColor(cv2.imread(str(APP_ROOT / 'assets/hdri/sunset.exr'), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB),
                dtype=torch.float32, device='cuda'
            )),
            'courtyard': EnvMap(torch.tensor(
                cv2.cvtColor(cv2.imread(str(APP_ROOT / 'assets/hdri/courtyard.exr'), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB),
                dtype=torch.float32, device='cuda'
            )),
        }
        pipeline = loaded_pipeline
        envmap = loaded_envmap


def start_session(req: gr.Request):
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    os.makedirs(user_dir, exist_ok=True)
    
    
def end_session(req: gr.Request):
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    shutil.rmtree(user_dir, ignore_errors=True)


def preprocess_image(image: Image.Image) -> Image.Image:
    """
    Preprocess the input image.

    Args:
        image (Image.Image): The input image.

    Returns:
        Image.Image: The preprocessed image.
    """
    if image is None:
        raise gr.Error("Upload an image first.")
    initialize_runtime()
    processed_image = pipeline.preprocess_image(image)
    return processed_image


def pack_state(latents: Tuple[SparseTensor, SparseTensor, int]) -> dict:
    shape_slat, tex_slat, res = latents
    return {
        'shape_slat_feats': shape_slat.feats.cpu().numpy(),
        'tex_slat_feats': tex_slat.feats.cpu().numpy(),
        'coords': shape_slat.coords.cpu().numpy(),
        'res': res,
    }
    
    
def unpack_state(state: dict) -> Tuple[SparseTensor, SparseTensor, int]:
    shape_slat = SparseTensor(
        feats=torch.from_numpy(state['shape_slat_feats']).cuda(),
        coords=torch.from_numpy(state['coords']).cuda(),
    )
    tex_slat = shape_slat.replace(torch.from_numpy(state['tex_slat_feats']).cuda())
    return shape_slat, tex_slat, state['res']


def get_seed(randomize_seed: bool, seed: int) -> int:
    """
    Get the random seed.
    """
    return np.random.randint(0, MAX_SEED) if randomize_seed else seed


def image_to_3d(
    image: Image.Image,
    seed: int,
    resolution: str,
    ss_guidance_strength: float,
    ss_guidance_rescale: float,
    ss_sampling_steps: int,
    ss_rescale_t: float,
    shape_slat_guidance_strength: float,
    shape_slat_guidance_rescale: float,
    shape_slat_sampling_steps: int,
    shape_slat_rescale_t: float,
    tex_slat_guidance_strength: float,
    tex_slat_guidance_rescale: float,
    tex_slat_sampling_steps: int,
    tex_slat_rescale_t: float,
    req: gr.Request,
    progress=gr.Progress(track_tqdm=True),
) -> str:
    if image is None:
        raise gr.Error("Upload an image first.")
    initialize_runtime()
    # --- Sampling ---
    try:
        outputs, latents = pipeline.run(
            image,
            seed=seed,
            preprocess_image=False,
            sparse_structure_sampler_params={
                "steps": ss_sampling_steps,
                "guidance_strength": ss_guidance_strength,
                "guidance_rescale": ss_guidance_rescale,
                "rescale_t": ss_rescale_t,
            },
            shape_slat_sampler_params={
                "steps": shape_slat_sampling_steps,
                "guidance_strength": shape_slat_guidance_strength,
                "guidance_rescale": shape_slat_guidance_rescale,
                "rescale_t": shape_slat_rescale_t,
            },
            tex_slat_sampler_params={
                "steps": tex_slat_sampling_steps,
                "guidance_strength": tex_slat_guidance_strength,
                "guidance_rescale": tex_slat_guidance_rescale,
                "rescale_t": tex_slat_rescale_t,
            },
            pipeline_type={
                "512": "512",
                "1024": "1024_cascade",
                "1536": "1536_cascade",
            }[resolution],
            return_latent=True,
        )
    except torch.OutOfMemoryError as exc:
        torch.cuda.empty_cache()
        raise gr.Error("The RTX 4090 ran out of VRAM. Try 512 resolution or close other GPU applications.") from exc
    mesh = outputs[0]
    mesh.simplify(16777216) # nvdiffrast limit
    images = render_utils.render_snapshot(mesh, resolution=1024, r=2, fov=36, nviews=STEPS, envmap=envmap)
    state = pack_state(latents)
    torch.cuda.empty_cache()
    
    # --- HTML Construction ---
    # The Stack of 48 Images
    images_html = ""
    for m_idx, mode in enumerate(MODES):
        for s_idx in range(STEPS):
            # ID Naming Convention: view-m{mode}-s{step}
            unique_id = f"view-m{m_idx}-s{s_idx}"
            
            # Logic: Only Mode 0, Step 0 is visible initially
            is_visible = (m_idx == DEFAULT_MODE and s_idx == DEFAULT_STEP)
            vis_class = "visible" if is_visible else ""
            
            # Image Source
            img_base64 = image_to_base64(Image.fromarray(images[mode['render_key']][s_idx]))
            
            # Render the Tag
            images_html += f"""
                <img id="{unique_id}" 
                     class="previewer-main-image {vis_class}" 
                     src="{img_base64}" 
                     loading="eager">
            """
    
    # Button Row HTML
    btns_html = ""
    for idx, mode in enumerate(MODES):        
        active_class = "active" if idx == DEFAULT_MODE else ""
        # Note: onclick calls the JS function defined in Head
        btns_html += f"""
            <img src="{mode['icon_base64']}" 
                 class="mode-btn {active_class}" 
                 onclick="selectMode({idx})"
                 title="{mode['name']}">
        """
    
    # Assemble the full component
    full_html = f"""
    <div class="previewer-container">
        <div class="tips-wrapper">
            <div class="tips-icon">💡Tips</div>
            <div class="tips-text">
                <p>● <b>Render Mode</b> - Click on the circular buttons to switch between different render modes.</p>
                <p>● <b>View Angle</b> - Drag the slider to change the view angle.</p>
            </div>
        </div>
        
        <!-- Row 1: Viewport containing 48 static <img> tags -->
        <div class="display-row">
            {images_html}
        </div>
        
        <!-- Row 2 -->
        <div class="mode-row" id="btn-group">
            {btns_html}
        </div>

        <!-- Row 3: Slider -->
        <div class="slider-row">
            <input type="range" id="custom-slider" min="0" max="{STEPS - 1}" value="{DEFAULT_STEP}" step="1" oninput="onSliderChange(this.value)">
        </div>
    </div>
    """
    
    return state, full_html


def extract_glb(
    state: dict,
    decimation_target: int,
    texture_size: int,
    req: gr.Request,
    progress=gr.Progress(track_tqdm=True),
) -> Tuple[str, str]:
    """
    Extract a GLB file from the 3D model.

    Args:
        state (dict): The state of the generated 3D model.
        decimation_target (int): The target face count for decimation.
        texture_size (int): The texture resolution.

    Returns:
        str: The path to the extracted GLB file.
    """
    if not state:
        raise gr.Error("Generate a 3D asset before extracting the GLB.")
    initialize_runtime()
    user_dir = str(PORTABLE_PATHS['outputs'])
    shape_slat, tex_slat, res = unpack_state(state)
    mesh = pipeline.decode_latent(shape_slat, tex_slat, res)[0]
    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices,
        faces=mesh.faces,
        attr_volume=mesh.attrs,
        coords=mesh.coords,
        attr_layout=pipeline.pbr_attr_layout,
        grid_size=res,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=decimation_target,
        texture_size=texture_size,
        remesh=True,
        remesh_band=1,
        remesh_project=0,
        use_tqdm=True,
    )
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%dT%H%M%S") + f".{now.microsecond // 1000:03d}"
    os.makedirs(user_dir, exist_ok=True)
    glb_path = os.path.join(user_dir, f'sample_{timestamp}.glb')
    glb.export(glb_path, extension_webp=True)
    torch.cuda.empty_cache()
    return glb_path, glb_path


with gr.Blocks(delete_cache=(600, 600), title="TRELLIS.2 Image to 3D") as demo:
    with gr.Row():
        with gr.Column(scale=1, min_width=360):
            image_prompt = gr.Image(label="Image Prompt", format="png", image_mode="RGBA", type="pil", height=400)
            
            resolution = gr.Radio(["512", "1024", "1536"], label="Resolution", value="1024")
            seed = gr.Slider(0, MAX_SEED, label="Seed", value=0, step=1)
            randomize_seed = gr.Checkbox(label="Randomize Seed", value=True)
            decimation_target = gr.Slider(100000, 500000, label="Decimation Target", value=300000, step=10000)
            texture_size = gr.Slider(1024, 4096, label="Texture Size", value=2048, step=1024)
            
            generate_btn = gr.Button("Generate")
                
            with gr.Accordion(label="Advanced Settings", open=False):                
                gr.Markdown("Stage 1: Sparse Structure Generation")
                with gr.Row():
                    ss_guidance_strength = gr.Slider(1.0, 10.0, label="Guidance Strength", value=7.5, step=0.1)
                    ss_guidance_rescale = gr.Slider(0.0, 1.0, label="Guidance Rescale", value=0.7, step=0.01)
                    ss_sampling_steps = gr.Slider(1, 50, label="Sampling Steps", value=12, step=1)
                    ss_rescale_t = gr.Slider(1.0, 6.0, label="Rescale T", value=5.0, step=0.1)
                gr.Markdown("Stage 2: Shape Generation")
                with gr.Row():
                    shape_slat_guidance_strength = gr.Slider(1.0, 10.0, label="Guidance Strength", value=7.5, step=0.1)
                    shape_slat_guidance_rescale = gr.Slider(0.0, 1.0, label="Guidance Rescale", value=0.5, step=0.01)
                    shape_slat_sampling_steps = gr.Slider(1, 50, label="Sampling Steps", value=12, step=1)
                    shape_slat_rescale_t = gr.Slider(1.0, 6.0, label="Rescale T", value=3.0, step=0.1)
                gr.Markdown("Stage 3: Material Generation")
                with gr.Row():
                    tex_slat_guidance_strength = gr.Slider(1.0, 10.0, label="Guidance Strength", value=1.0, step=0.1)
                    tex_slat_guidance_rescale = gr.Slider(0.0, 1.0, label="Guidance Rescale", value=0.0, step=0.01)
                    tex_slat_sampling_steps = gr.Slider(1, 50, label="Sampling Steps", value=12, step=1)
                    tex_slat_rescale_t = gr.Slider(1.0, 6.0, label="Rescale T", value=3.0, step=0.1)                

        with gr.Column(scale=10):
            with gr.Walkthrough(selected=0) as walkthrough:
                with gr.Step("Preview", id=0):
                    preview_output = gr.HTML(empty_html, label="3D Asset Preview", show_label=True, container=True)
                    extract_btn = gr.Button("Extract GLB")
                with gr.Step("Extract", id=1):
                    glb_output = gr.Model3D(label="Extracted GLB", height=724, show_label=True, display_mode="solid", clear_color=(0.25, 0.25, 0.25, 1.0))
                    download_btn = gr.DownloadButton(label="Download GLB")
                    
    output_buf = gr.State()
    

    # Handlers
    demo.load(start_session)
    demo.unload(end_session)
    
    image_prompt.upload(
        preprocess_image,
        inputs=[image_prompt],
        outputs=[image_prompt],
        concurrency_limit=1,
        concurrency_id="gpu",
    )

    generate_btn.click(
        get_seed,
        inputs=[randomize_seed, seed],
        outputs=[seed],
    ).then(
        lambda: gr.Walkthrough(selected=0), outputs=walkthrough
    ).then(
        image_to_3d,
        inputs=[
            image_prompt, seed, resolution,
            ss_guidance_strength, ss_guidance_rescale, ss_sampling_steps, ss_rescale_t,
            shape_slat_guidance_strength, shape_slat_guidance_rescale, shape_slat_sampling_steps, shape_slat_rescale_t,
            tex_slat_guidance_strength, tex_slat_guidance_rescale, tex_slat_sampling_steps, tex_slat_rescale_t,
        ],
        outputs=[output_buf, preview_output],
        concurrency_limit=1,
        concurrency_id="gpu",
    )
    
    extract_btn.click(
        lambda: gr.Walkthrough(selected=1), outputs=walkthrough
    ).then(
        extract_glb,
        inputs=[output_buf, decimation_target, texture_size],
        outputs=[glb_output, download_btn],
        concurrency_limit=1,
        concurrency_id="gpu",
    )

demo.queue(default_concurrency_limit=1, max_size=4)

# Launch the Gradio app
if __name__ == "__main__":
    os.makedirs(TMP_DIR, exist_ok=True)
    initialize_ui_assets()
    demo.launch(css=css, head=head)
