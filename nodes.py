from nodes import SaveImage
import json
from PIL import Image
import numpy as np
from PIL.PngImagePlugin import PngInfo
from comfy.cli_args import args # type: ignore
import folder_paths # type: ignore
from folder_paths import get_filename_list # type: ignore
import comfy
import os
import re
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import psutil
import time
from typing import Any, Dict, List, Tuple, Optional
from collections import OrderedDict, defaultdict
import threading

class ControlNetSelector:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "control_net_name": (get_filename_list("controlnet"),)
            }
        }
    
    RETURN_TYPES = (folder_paths.get_filename_list("controlnet"), )
    RETURN_NAMES = ("control_net_name",)
    FUNCTION = "get_control_net_name"
    
    CATEGORY = 'Uber Comfy'

    def get_control_net_name(self, control_net_name):
        return (control_net_name,)

class ControlNetOptionalLoader:
    @classmethod
    def INPUT_TYPES(s):
        # Extending the file list with a 'None' option for manual selection
        return {"required": { "control_net_name": (["None"] + folder_paths.get_filename_list("controlnet"), )}}

    RETURN_TYPES = ("CONTROL_NET",)
    FUNCTION = "load_controlnet"

    CATEGORY = "Uber Comfy"

    def load_controlnet(self, control_net_name):
        # Only proceed if a control_net_name is provided and it is not 'None'
        if control_net_name and control_net_name != "None":
            controlnet_path = folder_paths.get_full_path("controlnet", control_net_name)
            controlnet = comfy.controlnet.load_controlnet(controlnet_path)
            return (controlnet,)
        # Return None or skip the operation if 'None' is selected or no input is provided
        return (None,)

class DiffusersSelector:
    CATEGORY = 'Uber Comfy'
    RETURN_TYPES = (folder_paths.get_folder_paths("diffusers"), )
    RETURN_NAMES = ("model_path",)
    FUNCTION = "select_model_path"

    @classmethod
    def INPUT_TYPES(cls):
        paths = []
        for search_path in folder_paths.get_folder_paths("diffusers"):
            if os.path.exists(search_path):
                for root, subdirs, files in os.walk(search_path, followlinks=True):
                    if "model_index.json" in files:
                        paths.append(os.path.relpath(root, start=search_path))
        return {"required": {"model_path": (paths,), }}

    def select_model_path(self, model_path):
        # This function simply returns the model path that was selected
        return (model_path,)

class SaveImageJPGNoMeta(SaveImage):
    @classmethod
    def INPUT_TYPES(s):
        output = {
            "required": {
                "images": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "ComfyUI"}),
                "quality": ("INT", {"default": 100, "min": 1, "max": 100, "step": 1}),
            },
        }

        return output

    CATEGORY = "Uber Comfy"
    RETURN_TYPES = ()
    FUNCTION = "suplex_save_images"

    def suplex_save_images(
        self,
        images,
        filename_prefix="ComfyUI",
        format="jpeg",
        quality=92,
    ):
        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = (
            folder_paths.get_save_image_path(
                filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0]
            )
        )
        results = list()
        for batch_number, image in enumerate(images):
            i = 255.0 * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
            file = f"{filename_with_batch_num}_{counter:05}_.{format}"
            results.append(
                {"filename": file, "subfolder": subfolder, "type": self.type}
            )
            counter += 1
        
        img.save(os.path.join(full_output_folder, file), quality=quality, optimize=True)
        return {"ui": {"images": results}}

class MultiInputVariableRewrite:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True}),
            },
            "optional": {
                "a": ("STRING", {"forceInput": True}),
                "b": ("STRING", {"forceInput": True}),
                "c": ("STRING", {"forceInput": True}),
                "d": ("STRING", {"forceInput": True}),
                "e": ("STRING", {"forceInput": True}),
                # ... add more up to 'z' if needed
            }
        }
   
    CATEGORY = "Uber Comfy"
    FUNCTION = "multicombinetext"
    RETURN_NAMES = ("TEXT",)
    RETURN_TYPES = ("STRING",)

    def multicombinetext(self, text="", **kwargs):
        for key, value in kwargs.items():
            if value:
                text = text.replace(f"{{{key}}}", value)
        return (text,)
    
class TextRegexOperations:
    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "text": ("STRING", {"multiline": True, "forceInput": True}),
                "num_operations": ("INT", {"default": 1, "min": 1, "max": 20, "step": 1}),
            },
            "optional": {}
        }
        
        # Create inputs in interleaved order (pattern_1, replacement_1, multiline_1, pattern_2, etc.)
        for i in range(1, 21):
            inputs["optional"][f"pattern_{i}"] = ("STRING", {"multiline": True})
            inputs["optional"][f"replacement_{i}"] = ("STRING", {"multiline": True})
            inputs["optional"][f"use_multiline_{i}"] = ("BOOLEAN", {"default": True})
        
        return inputs
    
    RETURN_TYPES = ("STRING",)
    FUNCTION = "process_text"
    CATEGORY = "Uber Comfy"

    def process_text(self, text, num_operations, **kwargs):
        processed_text = text
        
        for i in range(1, num_operations + 1):
            pattern = kwargs.get(f"pattern_{i}", "")
            replacement = kwargs.get(f"replacement_{i}", "")
            use_multiline = kwargs.get(f"use_multiline_{i}", True)
            
            if pattern:
                try:
                    flags = re.MULTILINE if use_multiline else 0
                    processed_text = re.sub(pattern, replacement, processed_text, flags=flags)
                except re.error as e:
                    print(f"Regex error in operation {i}: {str(e)}")
                    print(f"Pattern: {pattern}")
        
        return (processed_text,)
    
class VideoSegmentCalculator:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "duration": ("FLOAT", {
                    "default": 30.0, 
                    "min": 1.0, 
                    "max": 3600.0, 
                    "step": 1.0, 
                    "tooltip": "Duration of each segment in seconds"
                }),
                "frame_rate": ("FLOAT", {
                    "default": 25.0, 
                    "min": 1.0, 
                    "max": 120.0, 
                    "step": 0.1,
                    "tooltip": "Frame rate of the video"
                }),
                "index": ("INT", {
                    "default": 0, 
                    "min": 0, 
                    "max": 1000, 
                    "step": 1,
                    "tooltip": "Current segment index (0-based)"
                }),
            },
            "optional": {
                "overlap_frames": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 10,
                    "step": 1,
                    "tooltip": "Number of frames to overlap between segments"
                }),
                "precise_timing": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Use precise decimal timing for audio trimming"
                }),
            }
        }
    
    RETURN_TYPES = ("INT", "INT", "FLOAT", "FLOAT", "FLOAT")
    RETURN_NAMES = ("frame_load_cap", "skip_first_frames", "force_rate", "start_time", "end_time")
    FUNCTION = "calculate_segment"
    CATEGORY = "Uber Comfy"
    
    def calculate_segment(self, duration, frame_rate, index, overlap_frames=0, precise_timing=True):
        # Calculate the exact frames for the given duration
        exact_frames = duration * frame_rate
        
        # Calculate the number of frames in each segment (using ceiling to prevent gaps)
        frames_per_segment = math.ceil(exact_frames)
        
        # Calculate skip_first_frames based on index, with optional overlap
        if index == 0:
            skip_first_frames = 0
        else:
            skip_first_frames = index * frames_per_segment - overlap_frames
            # Ensure we don't go negative
            skip_first_frames = max(0, skip_first_frames)
        
        # Calculate start and end times based on exact frame positions
        start_time = skip_first_frames / frame_rate
        end_time = (skip_first_frames + frames_per_segment) / frame_rate
        
        # Round to 2 decimal places for audio timing if needed
        if not precise_timing:
            start_time = round(start_time, 2)
            end_time = round(end_time, 2)
        
        return (frames_per_segment, skip_first_frames, frame_rate, start_time, end_time)
    

class ModelSimilarityNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "base_model":   ("MODEL",),
            "target_model": ("MODEL",)}
        }

    RETURN_TYPES  = ("STRING",)
    RETURN_NAMES  = ("similarity_report",)
    FUNCTION      = "compare"
    CATEGORY      = "Uber Comfy"

    # ---------- helpers ----------
    @staticmethod
    def _cross(to_q, to_k, to_v, rnd):
        h, e = to_q.shape
        lq = nn.Linear(h, e, bias=False); lq.weight.copy_(to_q)
        lk = nn.Linear(h, e, bias=False); lk.weight.copy_(to_k)
        lv = nn.Linear(h, e, bias=False); lv.weight.copy_(to_v)
        return torch.einsum(
            "ik,jk->ik",
            torch.softmax(torch.einsum("ij,kj->ik", lq(rnd), lk(rnd)), dim=-1),
            lv(rnd)
        )

    @classmethod
    def _unwrap(cls, obj):
        if isinstance(obj, dict):               return obj
        if hasattr(obj, "get_weights"):         return obj.get_weights()
        if hasattr(obj, "state_dict"):          return obj.state_dict()
        for a in ("model", "original_model"):
            if hasattr(obj, a):                 return cls._unwrap(getattr(obj, a))
        raise TypeError("Cannot unwrap MODEL object")

    # ---------- main ----------
    def compare(self, base_model, target_model):
        b_sd = self._unwrap(base_model)
        t_sd = self._unwrap(target_model)
        torch.manual_seed(114514)

        rnd, b_attn, sims, common = {}, {}, [], []

        # prefixes to scan: input, middle, output
        scan = [
            ("diffusion_model.input_blocks",  5),   # indices 0-4
            ("diffusion_model.middle_block", 1),    # index   0
            ("diffusion_model.output_blocks",11),   # indices 0-10
        ]

        # discover layers present in both models
        for prefix, max_idx in scan:
            for i in range(max_idx):
                key = f"{prefix}.{i}.1.transformer_blocks.0.attn1.to_q.weight"
                if key in b_sd and key in t_sd:
                    common.append((prefix, i))

        if not common:
            return ("No matching attention layers found in both models.",)

        # compute attention outputs for base model
        for p, i in common:
            q = b_sd[f"{p}.{i}.1.transformer_blocks.0.attn1.to_q.weight"]
            k = b_sd[f"{p}.{i}.1.transformer_blocks.0.attn1.to_k.weight"]
            v = b_sd[f"{p}.{i}.1.transformer_blocks.0.attn1.to_v.weight"]
            h, e = q.shape
            key = f"{p}.{i}"
            rnd[key]   = torch.randn(e, h)
            b_attn[key] = self._cross(q, k, v, rnd[key])

        # compare with target model
        for p, i in common:
            key = f"{p}.{i}"
            t_attn = self._cross(
                t_sd[f"{p}.{i}.1.transformer_blocks.0.attn1.to_q.weight"],
                t_sd[f"{p}.{i}.1.transformer_blocks.0.attn1.to_k.weight"],
                t_sd[f"{p}.{i}.1.transformer_blocks.0.attn1.to_v.weight"],
                rnd[key]
            )
            sims.append(torch.mean(torch.cosine_similarity(b_attn[key], t_attn)))

        score = torch.mean(torch.stack(sims)) * 100
        return (f"Similarity: {score:.2f}%  (compared {len(common)} blocks)",)
    
class ModelWeightDumperNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "show_shapes": ("BOOLEAN", {"default": True}),
            "filter_prefix": ("STRING", {"default": ""}),
        }}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("weight_info",)
    FUNCTION = "dump_weights"
    CATEGORY = "Uber Comfy"

    @classmethod
    def _unwrap(cls, obj):
        if isinstance(obj, dict):               return obj
        if hasattr(obj, "get_weights"):         return obj.get_weights()
        if hasattr(obj, "state_dict"):          return obj.state_dict()
        for a in ("model", "original_model"):
            if hasattr(obj, a):                 return cls._unwrap(getattr(obj, a))
        raise TypeError("Cannot unwrap MODEL object")

    def dump_weights(self, model, show_shapes=True, filter_prefix=""):
        sd = self._unwrap(model)
        
        output_lines = []
        output_lines.append(f"Total weights: {len(sd)}\n")
        output_lines.append("="*80 + "\n")
        
        # Filter keys if prefix is provided
        keys = sorted(sd.keys())
        if filter_prefix:
            keys = [k for k in keys if k.startswith(filter_prefix)]
            output_lines.append(f"Filtered by prefix: '{filter_prefix}'\n")
            output_lines.append(f"Matching weights: {len(keys)}\n")
            output_lines.append("="*80 + "\n")
        
        for key in keys:
            weight = sd[key]
            if show_shapes:
                shape_str = f" → {tuple(weight.shape)}" if hasattr(weight, 'shape') else ""
                dtype_str = f" ({weight.dtype})" if hasattr(weight, 'dtype') else ""
                output_lines.append(f"{key}{shape_str}{dtype_str}\n")
            else:
                output_lines.append(f"{key}\n")
        
        return ("".join(output_lines),)
    
class RunwareResolutionCalculator:
    """ComfyUI node for intelligent resolution calculation with model-specific presets."""
    
    PRESETS = {
        "Nano Banana 2": [
            # Square formats (1:1)
            (1024, 1024), (2048, 2048), (4096, 4096),
            # Landscape 3:2 ratio
            (1200, 896), (2400, 1792), (4800, 3584),
            # Portrait 2:3 ratio
            (896, 1200), (1792, 2400), (3584, 4800),
            # Landscape 5:4 ratio
            (1152, 928), (2304, 1856), (4608, 3712),
            # Portrait 4:5 ratio
            (928, 1152), (1856, 2304), (3712, 4608),
            # Landscape 3:2 wider
            (1264, 848), (2528, 1696), (5056, 3392), (5096, 3392),
            # Portrait 2:3 taller
            (848, 1264), (1696, 2528), (3392, 5056), (3392, 5096),
            # Ultra-wide landscape
            (1376, 768), (2752, 1536), (5504, 3072),
            # Ultra-tall portrait
            (768, 1376), (1536, 2752), (3072, 5504),
            # Cinematic ultra-wide
            (1548, 672), (1584, 672), (3168, 1344), (6336, 2688),
        ],
    }
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "full_image": ("IMAGE",),
                "model_preset": (list(cls.PRESETS.keys()), {
                    "default": "Nano Banana 2"
                }),
            },
            "optional": {
                "cropped_image": ("IMAGE",),
                "mask": ("MASK",),
            }
        }
    
    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("width", "height")
    FUNCTION = "calculate"
    CATEGORY = "Uber Comfy"
    
    def calculate(self, full_image, model_preset, cropped_image=None, mask=None):
        """Main calculation entry point."""
        _, full_h, full_w, _ = full_image.shape
        resolutions = self.PRESETS.get(model_preset, self.PRESETS["Nano Banana 2"])
        
        if cropped_image is None or mask is None:
            return self._direct_resolution_selection(full_w, full_h, resolutions, model_preset)
        
        return self._mask_constrained_selection(full_image, mask, resolutions, model_preset)
    
    def _direct_resolution_selection(self, img_w, img_h, resolutions, preset_name):
        """Select resolution using multi-tier filtering and ranking."""
        print(f"🎯 {preset_name} Optimizer (Direct Resolution Mode)")
        print(f"   Source dimensions: {img_w}×{img_h}")
        
        source_ratio = img_w / img_h
        source_area = img_w * img_h
        
        # Tier 1: Spatial filtering
        spatial_candidates = [
            (w, h) for w, h in resolutions 
            if w <= img_w and h <= img_h
        ]
        
        if not spatial_candidates:
            print(f"   ⚠️ No candidates within bounds, using 1024x1024")
            return (1024, 1024)
        
        # Tier 2: Aspect ratio bucketing
        def get_aspect_bucket(ratio):
            if 0.95 <= ratio <= 1.05:
                return "square"
            elif ratio > 1.5:
                return "wide"
            elif ratio < 0.67:
                return "tall"
            else:
                return "standard"
        
        source_bucket = get_aspect_bucket(source_ratio)
        
        # Tier 3: Multi-criteria scoring
        scored_candidates = []
        
        for res_w, res_h in spatial_candidates:
            res_ratio = res_w / res_h
            res_area = res_w * res_h
            res_bucket = get_aspect_bucket(res_ratio)
            
            log_ratio_distance = abs(np.log2(source_ratio) - np.log2(res_ratio))
            aspect_score = 1.0 / (1.0 + log_ratio_distance)
            
            utilization = res_area / source_area
            utilization_score = utilization
            
            bucket_bonus = 1.2 if res_bucket == source_bucket else 1.0
            
            quality_tier = min(1.0, np.log2(res_area / 1048576) / 2)
            
            # Weighted product
            composite = (
                (aspect_score ** 2.8) * 
                (utilization_score ** 1.5) * 
                bucket_bonus * 
                (1.0 + quality_tier * 0.3)
            )
            
            scored_candidates.append({
                'resolution': (res_w, res_h),
                'score': composite,
                'aspect_score': aspect_score,
                'utilization': utilization
            })
        
        # Maximize score
        best = max(scored_candidates, key=lambda x: x['score'])
        
        print(f"   ✓ Selected: {best['resolution'][0]}×{best['resolution'][1]} "
              f"(aspect: {best['aspect_score']:.3f}, util: {best['utilization']:.3f})")
        return best['resolution']
    
    def _mask_constrained_selection(self, full_image, mask_tensor, resolutions, preset_name):
        """Select resolution using bi-directional fitting analysis."""
        print(f"🎯 {preset_name} Optimizer (Mask-Based Mode)")
        
        _, full_h, full_w, _ = full_image.shape
        
        mask_data = mask_tensor.cpu().numpy()[0]
        active_pixels = mask_data > 0.5
        
        y_coords, x_coords = np.where(active_pixels)
        
        if len(y_coords) == 0 or len(x_coords) == 0:
            return (1024, 1024)
        
        roi_x1, roi_x2 = x_coords.min(), x_coords.max()
        roi_y1, roi_y2 = y_coords.min(), y_coords.max()
        roi_width = roi_x2 - roi_x1 + 1
        roi_height = roi_y2 - roi_y1 + 1
        
        print(f"   ROI bounds: {roi_width}×{roi_height}")
        
        roi_ratio = roi_width / roi_height
        roi_area = roi_width * roi_height
        
        scored_candidates = []
        
        for res_w, res_h in resolutions:
            target_ratio = res_w / res_h
            
            # Three fitting strategies
            fit_a_w = roi_width
            fit_a_h = int(roi_width / target_ratio)
            
            fit_b_h = roi_height
            fit_b_w = int(roi_height * target_ratio)
            
            scale_to_ratio = (target_ratio / roi_ratio) ** 0.5
            fit_c_w = int(roi_width * scale_to_ratio)
            fit_c_h = int(roi_height / scale_to_ratio)
            
            viable_fits = []
            
            for fit_w, fit_h, strategy in [
                (fit_a_w, fit_a_h, 'width_anchor'),
                (fit_b_w, fit_b_h, 'height_anchor'),
                (fit_c_w, fit_c_h, 'proportional')
            ]:
                if fit_w <= full_w and fit_h <= full_h:
                    expansion_w = fit_w - roi_width
                    expansion_h = fit_h - roi_height
                    total_expansion = expansion_w + expansion_h
                    
                    viable_fits.append({
                        'width': fit_w,
                        'height': fit_h,
                        'expansion': total_expansion,
                        'strategy': strategy
                    })
            
            if not viable_fits:
                continue
            
            best_fit = min(viable_fits, key=lambda x: x['expansion'])
            
            fitted_area = best_fit['width'] * best_fit['height']
            target_area = res_w * res_h
            
            expansion_ratio = fitted_area / roi_area
            expansion_efficiency = 1.0 / expansion_ratio
            
            scale_match = min(fitted_area, target_area) / max(fitted_area, target_area)
            
            strategy_weight = 1.1 if best_fit['strategy'] == 'proportional' else 1.0
            
            # Harmonic mean
            harmonic_mean = 3.0 / (
                (1.0 / expansion_efficiency) + 
                (1.0 / scale_match) + 
                (1.0 / strategy_weight)
            )
            
            scored_candidates.append({
                'resolution': (res_w, res_h),
                'score': harmonic_mean,
                'fitted': (best_fit['width'], best_fit['height']),
                'expansion': expansion_ratio,
                'strategy': best_fit['strategy']
            })
        
        if not scored_candidates:
            return (1024, 1024)
        
        best = max(scored_candidates, key=lambda x: x['score'])
        
        print(f"   Fitted area: {best['fitted'][0]}×{best['fitted'][1]} ({best['strategy']})")
        print(f"   ✓ Selected: {best['resolution'][0]}×{best['resolution'][1]} "
              f"(expansion: {best['expansion']:.2f}x)")
        return best['resolution']

class AdaptiveImageScaler:
    """Intelligent image scaling with optional ML-based upscaling and dimension constraints."""
    
    INTERPOLATION_MODES = [
        "lanczos", "bicubic", "bilinear", "area", "nearest-exact"
    ]
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "target_width": ("INT",),
                "target_height": ("INT",),
                "interpolation": (cls.INTERPOLATION_MODES,),
                "dimension_alignment": ("INT", {
                    "default": 0, "min": 0, "max": 256, "step": 1
                }),
            },
            "optional": {
                "upscale_model": ("UPSCALE_MODEL",),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = "Uber Comfy"
    
    MODEL_ENGAGEMENT_THRESHOLD = 1.08
    TILE_REDUCTION_FACTOR = 0.75
    MIN_SAFE_TILE_SIZE = 128
    
    def process(self, image, target_width, target_height, interpolation, 
                dimension_alignment, upscale_model=None):
        """Main processing pipeline."""
        import comfy.utils
        from comfy import model_management
        
        batch_size, source_h, source_w, channels = image.shape
        
        if dimension_alignment > 0:
            target_width = (target_width // dimension_alignment) * dimension_alignment
            target_height = (target_height // dimension_alignment) * dimension_alignment
        
        scale_w = target_width / source_w
        scale_h = target_height / source_h
        geometric_mean_scale = (scale_w * scale_h) ** 0.5
        
        print(f"🖼️  Adaptive Scaler: {source_w}×{source_h}×{channels} → {target_width}×{target_height}")
        print(f"    Scale: {geometric_mean_scale:.3f}x (w:{scale_w:.2f}, h:{scale_h:.2f}) | {interpolation}")
        
        use_ml_model = (
            geometric_mean_scale > self.MODEL_ENGAGEMENT_THRESHOLD and 
            upscale_model is not None
        )
        
        if use_ml_model:
            return self._ml_assisted_pipeline(
                image, target_width, target_height, interpolation, channels,
                upscale_model, comfy, model_management
            )
        else:
            return self._standard_interpolation_pipeline(
                image, target_width, target_height, interpolation, 
                geometric_mean_scale, comfy
            )
    
    def _ml_assisted_pipeline(self, image, target_w, target_h, interpolation, 
                              channels, model, comfy, model_mgmt):
        """ML-based upscaling with synchronized alpha processing."""
        print(f"    ⚡ ML upscaling engaged")
        
        has_alpha = (channels == 4)
        
        if has_alpha:
            print(f"    💎 Alpha channel: synchronized processing")
            rgb_channels = image[:, :, :, :3]
            alpha_channel = image[:, :, :, 3:4]
        else:
            rgb_channels = image
            alpha_channel = None
        
        device = model_mgmt.get_torch_device()
        model.to(device)
        
        rgb_gpu = rgb_channels.movedim(-1, -3).to(device)
        
        upscaled_rgb = self._progressive_tiled_upscale(
            rgb_gpu, model, comfy, model_mgmt
        )
        
        model.cpu()
        
        upscaled_rgb = torch.clamp(upscaled_rgb.movedim(-3, -1), 0.0, 1.0)
        
        upscaled_rgb = upscaled_rgb.movedim(-1, 1)
        upscaled_rgb = comfy.utils.common_upscale(
            upscaled_rgb, target_w, target_h, interpolation, "disabled"
        )
        upscaled_rgb = upscaled_rgb.movedim(1, -1)
        
        if has_alpha:
            # Bicubic for quality
            alpha_scaled = self._scale_alpha_channel(
                alpha_channel, target_w, target_h, use_bicubic=True
            )
            result = torch.cat([upscaled_rgb, alpha_scaled], dim=-1)
            print(f"    💎 Alpha reconstructed with bicubic interpolation")
        else:
            result = upscaled_rgb
        
        return (result,)
    
    def _progressive_tiled_upscale(self, tensor, model, comfy, model_mgmt):
        """Progressive tile reduction (0.75x vs 0.5x)."""
        current_tile = 512
        overlap = 32
        
        while True:
            try:
                steps = tensor.shape[0] * comfy.utils.get_tiled_scale_steps(
                    tensor.shape[3], tensor.shape[2],
                    tile_x=current_tile, tile_y=current_tile, overlap=overlap
                )
                
                pbar = comfy.utils.ProgressBar(steps)
                
                result = comfy.utils.tiled_scale(
                    tensor,
                    lambda t: model(t),
                    tile_x=current_tile, tile_y=current_tile,
                    overlap=overlap,
                    upscale_amount=model.scale,
                    pbar=pbar
                )
                
                return result
                
            except model_mgmt.OOM_EXCEPTION as e:
                new_tile = int(current_tile * self.TILE_REDUCTION_FACTOR)
                
                if new_tile < self.MIN_SAFE_TILE_SIZE:
                    print(f"    ❌ Tile size below safe threshold ({self.MIN_SAFE_TILE_SIZE}px)")
                    raise e
                
                print(f"    ⚠️  Memory limit reached, reducing tile: {current_tile}→{new_tile}px")
                current_tile = new_tile
    
    def _scale_alpha_channel(self, alpha, target_w, target_h, use_bicubic=True):
        """Scale alpha using squeeze/unsqueeze (different from permute)."""
        alpha_reshaped = alpha.squeeze(-1).unsqueeze(1)
        
        mode = 'bicubic' if use_bicubic else 'bilinear'
        
        alpha_scaled = F.interpolate(
            alpha_reshaped,
            size=(target_h, target_w),
            mode=mode,
            align_corners=False if mode == 'bilinear' else None
        )
        
        alpha_result = alpha_scaled.squeeze(1).unsqueeze(-1)
        
        return alpha_result
    
    def _standard_interpolation_pipeline(self, image, target_w, target_h, 
                                          method, scale, comfy):
        """Direct interpolation with integrated channels."""
        direction = "⬇️ downscale" if scale < 1.0 else "⬆️ upscale"
        print(f"    {direction} via {method} interpolation")
        
        tensor_bchw = image.movedim(-1, 1)
        scaled_bchw = comfy.utils.common_upscale(
            tensor_bchw, target_w, target_h, method, "disabled"
        )
        result = scaled_bchw.movedim(1, -1)
        
        return (result,)

NODE_CLASS_MAPPINGS = {
    "ControlNet Selector": ControlNetSelector,
    "ControlNetOptionalLoader": ControlNetOptionalLoader,
    "DiffusersSelector": DiffusersSelector,
    "SaveImageJPGNoMeta": SaveImageJPGNoMeta,
    "MultiInputVariableRewrite": MultiInputVariableRewrite,
    "TextRegexOperations": TextRegexOperations,
    "VideoSegmentCalculator": VideoSegmentCalculator,
    "ModelSimilarityNode": ModelSimilarityNode,
    "ModelWeightDumperNode": ModelWeightDumperNode,
    "RunwareResolutionCalculator": RunwareResolutionCalculator,
    "AdaptiveImageScaler": AdaptiveImageScaler,

}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ControlNet Selector": "ControlNet Selector",
    "ControlNetOptionalLoader": "Load Optional ControlNet Model",
    "DiffusersSelector": "Diffusers Selector",
    "SaveImageJPGNoMeta": "Save Image JPG No Meta",
    "MultiInputVariableRewrite": "Multi Input Variable Rewrite",
    "TextRegexOperations": "Text Regex Operations",
    "VideoSegmentCalculator": "Video Segment Calculator",
    "ModelSimilarityNode": "Model Similarity Node",
    "ModelWeightDumperNode": "Model Weight Dumper",
    "RunwareResolutionCalculator": "Runware Resolution Calculator",
    "AdaptiveImageScaler": "Adaptive Image Scaler",
}
