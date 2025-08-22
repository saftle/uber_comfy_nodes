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

# Export node
NODE_CLASS_MAPPINGS = {
    "ControlNet Selector": ControlNetSelector,
    "ControlNetOptionalLoader": ControlNetOptionalLoader,
    "DiffusersSelector": DiffusersSelector,
    "SaveImageJPGNoMeta": SaveImageJPGNoMeta,
    "MultiInputVariableRewrite": MultiInputVariableRewrite,
    "TextRegexOperations": TextRegexOperations,
    "VideoSegmentCalculator": VideoSegmentCalculator,
    "ModelSimilarityNode": ModelSimilarityNode,
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
}
