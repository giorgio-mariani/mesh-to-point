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
    """Configure the compositor node tree for a single view.

        The function clears any existing nodes, creates a ``RLayers`` node to
        access the rendered image data, and then adds output nodes for the
        passes requested in the :class:`ViewConfig`.

    Args:

        view : :class:`ViewConfig`
                Configuration describing which passes to generate.  The
                ``mask``, ``depth`` and ``normals`` attributes are ``Path``
                objects when the corresponding pass should be written to disk.
        tmp_dir : :class:`pathlib.Path`
                Directory where the output files will be stored.

    Notes
    -----

    * If ``view.mask`` is set, a ``Math`` node is inserted to threshold
        the alpha channel.  The resulting value is routed to a file
        output node that writes a single‑channel 8‑bit PNG.
    * ``view.depth`` and ``view.normals`` are passed directly to
        :func:`add_output`, which creates a ``CompositorNodeOutputFile``
        with the appropriate format settings.
    * The function does not return anything; it mutates the current
        scene's node tree.
    """

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
