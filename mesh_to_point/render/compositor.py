import bpy
from pathlib import Path
from .config import GlobalConfig


def add_output(
    input_node,
    output_path: Path,
    image_format: str = "OPEN_EXR_MULTILAYER",
    mode: str = "RGB",
    depth: str = "32",
    data_type: str = "FLOAT",
):
    data_name = str(output_path.name)
    compositor_tree = bpy.context.scene.compositing_node_group
    out = compositor_tree.nodes.new("CompositorNodeOutputFile")
    out.format.file_format = image_format
    out.format.color_mode = mode
    out.format.color_depth = depth
    out.directory = str(output_path.parent.absolute())
    out.file_name = data_name

    out.file_output_items.new(data_type, "")
    compositor_tree.links.new(input_node, out.inputs[""])
    return out


def setup_compositor(cfg: GlobalConfig, tmp_dir: Path):
    scene = bpy.context.scene

    # Create compositor tree
    comp_tree = bpy.data.node_groups.new("Compositor Tree", "CompositorNodeTree")
    scene.compositing_node_group = comp_tree

    render_layers = comp_tree.nodes.new(type="CompositorNodeRLayers")
    output = comp_tree.nodes.new(type="NodeGroupOutput")

    comp_tree.interface.new_socket(
        name="Image", in_out="OUTPUT", socket_type="NodeSocketColor"
    )
    comp_tree.links.new(output.inputs["Image"], render_layers.outputs["Image"])

    if cfg.depth_pass:
        add_output(render_layers.outputs["Depth"], tmp_dir / "depth")
