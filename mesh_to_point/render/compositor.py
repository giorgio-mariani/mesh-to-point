import bpy
from pathlib import Path
from .config import ViewConfig


def clear_nodes():
    tree = bpy.context.scene.node_tree
    for node in tree.nodes:
        tree.nodes.remove(node)


def add_output(node, path: Path, fmt="OPEN_EXR", mode="RGB", depth="16"):
    out = bpy.context.scene.node_tree.nodes.new("CompositorNodeOutputFile")
    out.format.file_format = fmt
    out.format.color_mode = mode
    out.format.color_depth = depth
    out.base_path = str(path)
    return out


def setup_compositor(view: ViewConfig, tmp_dir: Path):
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

    clear_nodes()
    tree = bpy.context.scene.node_tree
    rl = tree.nodes.new("CompositorNodeRLayers")

    if view.mask:
        math = tree.nodes.new("CompositorNodeMath")
        math.operation = "GREATER_THAN"
        tree.links.new(rl.outputs["Alpha"], math.inputs[0])
        math.inputs[1].default_value = 0.0001
        add_output(
            math.outputs["Value"], tmp_dir / "alpha", fmt="PNG", mode="BW", depth="8"
        )

    if view.depth:
        add_output(rl.outputs["Depth"], tmp_dir / "depth")

    if view.normals:
        add_output(rl.outputs["Normal"], tmp_dir / "normal")
