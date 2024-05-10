Misc ComfyUI Nodes. Will continue adding more as I find nodes that don't otherwise exist.

1. ControlNet Selector
   This will let yous select from all of your ControlNet models without forcing it to load them. Great for passing it to other nodes.
2. Load Optional ControlNet Model
   This is a fork of the original "Load ControlNet Model" node in comfy core, but instead of it auto-assigning a ControlNet load and forcefully loading it, it has an extra None value that you can use to trigger it to no longer load anything.
