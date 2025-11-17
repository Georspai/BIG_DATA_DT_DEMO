from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional

import pyvista as pv


def export_plotter_png(
    plotter: pv.Plotter,
    png_path: str | Path,
    view: str = "iso",
    parallel_projection: bool = False,
    camera_zoom: float = 1.25,
    **screenshot_kwargs: Any,
) -> str:
    """
    Export a static PNG from the current plotter view.

    Parameters
    ----------
    plotter : pv.Plotter
        Plotter with all actors already added.
    png_path : str or Path
        Output file path. Extension will be forced to `.png`.
    view : {"iso", "xy", "xz", "yz"}, optional
        Camera preset to use before taking the screenshot.
    parallel_projection : bool, optional
        If True, use orthographic projection, otherwise perspective.
    camera_zoom : float, optional
        Multiplicative zoom factor (>1 zooms in, <1 zooms out).
    **screenshot_kwargs :
        Extra kwargs forwarded to `plotter.screenshot()`.

    Returns
    -------
    str
        Path of the written PNG.
    """
    png_path = Path(png_path).with_suffix(".png")

    # set a reasonable view
    if view == "iso":
        plotter.view_isometric()
    elif view == "xy":
        plotter.view_xy()
    elif view == "xz":
        plotter.view_xz()
    elif view == "yz":
        plotter.view_yz()

    # switch projection mode based on flag
    if parallel_projection:
        plotter.enable_parallel_projection()
    else:
        plotter.disable_parallel_projection()

    if camera_zoom and camera_zoom != 1.0:
        plotter.camera.Zoom(camera_zoom)

    # this triggers a render internally if needed
    plotter.screenshot(str(png_path), **screenshot_kwargs)

    return str(png_path)


def export_plotter_gif(
    plotter: pv.Plotter,
    gif_path: str | Path,
    n_frames: int = 36,
    view: str = "iso",
    parallel_projection: bool = False,
    fps: int = 5,
    camera_zoom: float = 1.25,
) -> str:
    """
    Export a 360° rotating GIF of the current scene (perspective by default).

    Parameters
    ----------
    plotter : pv.Plotter
        Plotter with all actors already added.
    gif_path : str or Path
        Output file path. Extension will be forced to `.gif`.
    n_frames : int, optional
        Number of frames for the full 360° rotation.
    view : {"iso", "xy", "xz", "yz"}, optional
        Initial camera preset.
    parallel_projection : bool, optional
        If True, use orthographic projection, otherwise perspective.
    fps : int, optional
        Frames per second for the GIF animation (lower = slower playback).
    camera_zoom : float, optional
        Multiplicative zoom factor (>1 zooms in, <1 zooms out).

    Returns
    -------
    str
        Path of the written GIF.
    """
    gif_path = Path(gif_path).with_suffix(".gif")

    # initial view
    if view == "iso":
        plotter.view_isometric()
    elif view == "xy":
        plotter.view_xy()
    elif view == "xz":
        plotter.view_xz()
    elif view == "yz":
        plotter.view_yz()

    if parallel_projection:
        plotter.enable_parallel_projection()
    else:
        plotter.disable_parallel_projection()

    if camera_zoom and camera_zoom != 1.0:
        plotter.camera.Zoom(camera_zoom)

    # open gif writer
    plotter.open_gif(str(gif_path), fps=fps)

    # rotate around the vertical axis
    step = 360.0 / n_frames
    for _ in range(n_frames):
        # VTK camera azimuth rotation
        plotter.camera.Azimuth(step)
        plotter.render()
        plotter.write_frame()

    
    

    return str(gif_path)


def export_plotter(
    plotter: pv.Plotter,
    filename: str,
    export_static_image: bool = True,
    export_gif: bool = False,
    export_file: bool = True,
    n_gif_frames: int = 36,
    gif_fps: int = 5,
    camera_zoom: float = 1.25,
) -> Dict[str, str]:
    """
    Export a PyVista plotter scene to:
      - a geometry/scene file (`export_file`)
      - a PNG screenshot (`export_static_image`)
      - a rotating GIF (`export_gif`)

    Parameters
    ----------
    plotter : pv.Plotter
        Plotter with all actors already added.
    filename : str
        Target filename for the scene export. The basename (without extension)
        is re-used for the PNG and GIF.
        Example: "scene.obj" → "scene.obj", "scene.png", "scene.gif".
    export_static_image : bool, optional
        If True, export a perspective PNG from an isometric view.
    export_gif : bool, optional
        If True, export a 360° rotating GIF.
    export_file : bool, optional
        If True, export the scene/geometry according to the extension.
        Currently supports `.obj` and `.vtkjs` from the plotter.
    n_gif_frames : int, optional
        Number of frames for the GIF rotation.
    gif_fps : int, optional
        Playback speed of the GIF (frames per second).
    camera_zoom : float, optional
        Multiplicative zoom factor applied before captures.

    Returns
    -------
    dict
        Dictionary with keys in {"file", "png", "gif"} mapping to written paths.
    """
    path = Path(filename)
    base = path.with_suffix("")  # without extension
    outputs: Dict[str, str] = {}

    # 1) Export scene / geometry
    if export_file:
        ext = path.suffix.lower()
        if ext == ".obj":
            # Exports the full rendered scene as OBJ
            plotter.export_obj(str(path))
        elif ext == ".vtkjs":
            # Web-friendly VTKJS scene
            plotter.export_vtksz(str(path))
        else:
            raise ValueError(
                f"Scene export for extension '{ext}' is not supported. "
                "Use '.obj' or '.vtkjs', or export meshes individually."
            )
        outputs["file"] = str(path)

    # 2) Export static PNG (perspective view)
    if export_static_image:
        png_path = base.with_suffix(".png")
        outputs["png"] = export_plotter_png(
            plotter,
            png_path,
            view="iso",
            parallel_projection=False,  # perspective
            camera_zoom=camera_zoom,
        )

    # 3) Export rotating GIF (perspective view)
    if export_gif:
        gif_path = base.with_suffix(".gif")
        outputs["gif"] = export_plotter_gif(
            plotter,
            gif_path,
            n_frames=n_gif_frames,
            view="iso",
            parallel_projection=False,  # perspective
            fps=gif_fps,
            camera_zoom=camera_zoom,
        )

    return outputs
