import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from PIL import Image, ImageChops, ImageDraw

from .io import save_pyvista_figure


def set_up_camera(plotter, center, camera_distance=300000, upshift=30000):
    center = np.array([center[0], center[1] - upshift, center[2]])

    plotter.camera.up = (0, -1, 0)
    plotter.camera.position = center + np.array(
        [camera_distance, -camera_distance / 5, camera_distance]
    )
    plotter.camera.focal_point = center
    plotter.enable_fly_to_right_click()

    return plotter


def crop_to_circle(img_path, output_path, border_width=10, preserve_alpha=True):
    img = Image.open(img_path).convert("RGBA")

    # create a circular mask
    mask = Image.new("L", img.size, 0)  # 'L' for grayscale
    draw = ImageDraw.Draw(mask)
    draw.ellipse((0, 0, img.size[0], img.size[1]), fill=255)

    if preserve_alpha:
        # combine with existing alpha (if any)
        existing_alpha = img.getchannel("A")
        combined_mask = ImageChops.multiply(existing_alpha, mask)
    else:
        combined_mask = mask

    # apply the mask to the image
    img.putalpha(combined_mask)

    if border_width is not None and border_width > 0:
        # add a black border around the circle
        border = Image.new("RGBA", img.size, (0, 0, 0, 0))
        border_draw = ImageDraw.Draw(border)
        border_draw.ellipse(
            (0, 0, img.size[0], img.size[1]),
            outline="black",
            width=border_width,
        )
        img = Image.alpha_composite(img, border)

    img.save(output_path, "PNG")


def get_camera_vector(cpos):
    camera_vector = np.array(cpos[1]) - np.array(cpos[0])
    camera_vector /= np.linalg.norm(camera_vector)
    return camera_vector


def plot_zoom_indicator(
    plotter,
    zoom_cpos,
    wide_cpos,
    radius=7500,
    shift=10000,
    height=10000,
    resolution=50,
    line_width=25,
):
    camera_vector = np.array(zoom_cpos[1]) - np.array(wide_cpos[0])
    camera_vector /= np.linalg.norm(camera_vector)

    # add a hollow cylinder facing the camera at the zoom point
    cylinder = pv.Cylinder(
        center=zoom_cpos[1] - camera_vector * shift,
        direction=camera_vector,
        radius=radius,
        height=height,
        resolution=resolution,
    )
    plotter.add_mesh(
        cylinder,
        color="black",
        opacity=1,
        style="wireframe",
        line_width=line_width,
    )


def render(
    plotter,
    name,
    figure_path,
    circle_crop=False,
    interactive=False,
    scale=10,
    circle_crop_border_width=80,
):
    if interactive:
        plotter.enable_fly_to_right_click()
        plotter.show()
    else:
        save_pyvista_figure(
            plotter, name, figure_path, formats=["png"], scale=scale, show=True
        )
        if circle_crop:
            input_path = figure_path / f"{name}.png"
            output_path = figure_path / f"{name}_circular.png"
            crop_to_circle(
                input_path,
                output_path,
                border_width=circle_crop_border_width,
                preserve_alpha=False,
            )
        plotter.close()


def make_composite_figure(
    name,
    figure_path,
    size=(6, 8),
    inset_bounds=(0.63, 0.55, 0.5, 0.5),
    flip_inset=False,
    zoom=True,
    dpi=500,
):
    fig, ax = plt.subplots(
        1, 1, figsize=size, dpi=dpi, gridspec_kw=dict(hspace=0, wspace=0)
    )

    # set background transparent
    ax.set_facecolor("none")
    fig.patch.set_alpha(0)

    img = plt.imread(figure_path / f"{name}.png")
    ax.imshow(img, interpolation="none", aspect=None)
    ax.axis("off")

    if zoom:
        sub_ax = ax.inset_axes(inset_bounds)
        sub_ax.set_facecolor("none")

        sub_img = plt.imread(figure_path / f"{name}_zoom_circular.png")
        if flip_inset:
            # reverse columns
            sub_img = sub_img[:, ::-1, :]
        sub_ax.imshow(sub_img, aspect=None)
        sub_ax.axis("off")
    else:
        sub_ax = None

    return fig, ax, sub_ax
