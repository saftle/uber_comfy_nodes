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
    
    CATEGORY = 'Suplex'

    def get_control_net_name(self, control_net_name):
        return (control_net_name,)

class ControlNetOptionalLoader:
    @classmethod
    def INPUT_TYPES(s):
        # Extending the file list with a 'None' option for manual selection
        return {"required": { "control_net_name": (["None"] + folder_paths.get_filename_list("controlnet"), )}}

    RETURN_TYPES = ("CONTROL_NET",)
    FUNCTION = "load_controlnet"

    CATEGORY = "Suplex"

    def load_controlnet(self, control_net_name):
        # Only proceed if a control_net_name is provided and it is not 'None'
        if control_net_name and control_net_name != "None":
            controlnet_path = folder_paths.get_full_path("controlnet", control_net_name)
            controlnet = comfy.controlnet.load_controlnet(controlnet_path)
            return (controlnet,)
        # Return None or skip the operation if 'None' is selected or no input is provided
        return (None,)

class DiffusersSelector:
    CATEGORY = 'Suplex'
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

    CATEGORY = "Suplex"
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
   
    CATEGORY = "Suplex"
    FUNCTION = "multicombinetext"
    RETURN_NAMES = ("TEXT",)
    RETURN_TYPES = ("STRING",)

    def multicombinetext(self, text="", **kwargs):
        for key, value in kwargs.items():
            if value:
                text = text.replace(f"{{{key}}}", value)
        return (text,)

# Export node
NODE_CLASS_MAPPINGS = {
    "ControlNet Selector": ControlNetSelector,
    "ControlNetOptionalLoader": ControlNetOptionalLoader,
    "DiffusersSelector": DiffusersSelector,
    "SaveImageJPGNoMeta": SaveImageJPGNoMeta,
    "MultiInputVariableRewrite": MultiInputVariableRewrite,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ControlNet Selector": "ControlNet Selector",
    "ControlNetOptionalLoader": "Load Optional ControlNet Model",
    "DiffusersSelector": "Diffusers Selector",
    "SaveImageJPGNoMeta": "Save Image JPG No Meta",
    "MultiInputVariableRewrite": "Multi Input Variable Rewrite",
}
