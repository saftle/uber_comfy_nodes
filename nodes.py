import folder_paths
from folder_paths import get_filename_list
import comfy

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

# Export node
NODE_CLASS_MAPPINGS = {
    "ControlNet Selector": ControlNetSelector,
    "ControlNetOptionalLoader": ControlNetOptionalLoader,
    "DiffusersSelector": DiffusersSelector,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ControlNet Selector": "ControlNet Selector",
    "ControlNetOptionalLoader": "Load Optional ControlNet Model",
    "DiffusersSelector": "Diffusers Selector",
}
