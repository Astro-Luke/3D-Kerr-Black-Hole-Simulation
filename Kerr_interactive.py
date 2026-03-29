import time
import numpy as np
import pyvista as pv
import pyvistaqt as pvqt

# ============================================================
# Kerr black hole - PyVista interactive animation
# Cutaway view for readability
# Units: G = c = M = 1
# ============================================================

# -----------------------------
# Parameters
# -----------------------------
N_THETA = 220
N_PHI = 220
DT = 0.03

COLOR_INNER = "#c43c39"   # red
COLOR_OUTER = "#2c6bed"   # blue
COLOR_ERGO  = "#bdbdbd"   # gray

OPACITY_INNER = 0.90
OPACITY_OUTER = 0.55
OPACITY_ERGO  = 0.22

# keep y >= 0 half
CLIP_NORMAL = (0, -1, 0)
CLIP_ORIGIN = (0, 0, 0)

# -----------------------------
# Kerr radii
# -----------------------------
def r_plus(a):
    return 1.0 + np.sqrt(np.maximum(0.0, 1.0 - a**2))

def r_minus(a):
    return 1.0 - np.sqrt(np.maximum(0.0, 1.0 - a**2))

def r_ergosphere_outer(a, theta):
    return 1.0 + np.sqrt(np.maximum(0.0, 1.0 - a**2 * np.cos(theta)**2))

# -----------------------------
# Smooth periodic parameter
# -----------------------------
def spin_parameter(t):
    return 0.5 * (1.0 - np.cos(t))

# -----------------------------
# Surface builder
# -----------------------------
theta = np.linspace(0.0, np.pi, N_THETA)
phi = np.linspace(0.0, 2.0 * np.pi, N_PHI)
TH, PH = np.meshgrid(theta, phi, indexing="ij")
DIMS = (N_THETA, N_PHI, 1)

def make_grid_from_r(r):
    rho = r * np.sin(TH)
    z = r * np.cos(TH)

    x = rho * np.cos(PH)
    y = rho * np.sin(PH)

    pts = np.column_stack((
        x.ravel(order="F"),
        y.ravel(order="F"),
        z.ravel(order="F")
    ))

    grid = pv.StructuredGrid()
    grid.points = pts
    grid.dimensions = DIMS
    return grid

def build_surfaces(a):
    r_out = np.full_like(TH, r_plus(a), dtype=float)
    outer = make_grid_from_r(r_out)

    r_ergo = r_ergosphere_outer(a, TH)
    ergo = make_grid_from_r(r_ergo)

    rin = r_minus(a)
    if rin > 1e-8:
        r_in = np.full_like(TH, rin, dtype=float)
        inner = make_grid_from_r(r_in)
    else:
        tiny = np.full_like(TH, 1e-6, dtype=float)
        inner = make_grid_from_r(tiny)

    # Clip to get cutaway view
    outer_c = outer.clip(normal=CLIP_NORMAL, origin=CLIP_ORIGIN, invert=False)
    ergo_c  = ergo.clip(normal=CLIP_NORMAL, origin=CLIP_ORIGIN, invert=False)
    inner_c = inner.clip(normal=CLIP_NORMAL, origin=CLIP_ORIGIN, invert=False)

    return outer_c, inner_c, ergo_c, rin

# -----------------------------
# Initial state
# -----------------------------
a0 = 0.0
outer_mesh, inner_mesh, ergo_mesh, rin0 = build_surfaces(a0)

plotter = pvqt.BackgroundPlotter(window_size=(1450, 950))
plotter.set_background("white")

# Depth peeling helps with transparency
try:
    plotter.enable_depth_peeling()
except Exception:
    pass

actor_ergo = plotter.add_mesh(
    ergo_mesh,
    color=COLOR_ERGO,
    opacity=OPACITY_ERGO,
    smooth_shading=True,
    show_edges=True,
    edge_color="#666666",
    line_width=0.8,
    name="ergosphere",
)

actor_outer = plotter.add_mesh(
    outer_mesh,
    color=COLOR_OUTER,
    opacity=OPACITY_OUTER,
    smooth_shading=True,
    show_edges=False,
    name="outer_horizon",
)

actor_inner = plotter.add_mesh(
    inner_mesh,
    color=COLOR_INNER,
    opacity=0.0,
    smooth_shading=True,
    show_edges=False,
    name="inner_horizon",
)

text_actor = plotter.add_text(
    f"Kerr black hole  a = {a0:.3f}",
    position="upper_edge",
    font_size=12,
    color="black",
)

plotter.show_axes()

# Better camera for cutaway
plotter.camera_position = [
    (5.5, -8.0, 2.8),
    (0.0,  0.0, 0.0),
    (0.0,  0.0, 1.0),
]

# -----------------------------
# Animation loop
# -----------------------------
t = 0.0
while plotter.app_window is not None:
    a = spin_parameter(t)

    new_outer, new_inner, new_ergo, rin = build_surfaces(a)

    plotter.remove_actor("ergosphere")
    plotter.remove_actor("outer_horizon")
    plotter.remove_actor("inner_horizon")

    actor_ergo = plotter.add_mesh(
        new_ergo,
        color=COLOR_ERGO,
        opacity=OPACITY_ERGO,
        smooth_shading=True,
        show_edges=True,
        edge_color="#666666",
        line_width=0.8,
        name="ergosphere",
        render=False,
    )

    actor_outer = plotter.add_mesh(
        new_outer,
        color=COLOR_OUTER,
        opacity=OPACITY_OUTER,
        smooth_shading=True,
        show_edges=False,
        name="outer_horizon",
        render=False,
    )

    actor_inner = plotter.add_mesh(
        new_inner,
        color=COLOR_INNER,
        opacity=(0.90 if rin > 1e-8 else 0.0),
        smooth_shading=True,
        show_edges=False,
        name="inner_horizon",
        render=False,
    )

    plotter.remove_actor(text_actor)
    text_actor = plotter.add_text(
        f"Kerr black hole  a = {a:.3f}",
        position="upper_edge",
        font_size=12,
        color="black",
    )

    plotter.render()
    plotter.app.processEvents()

    t += 0.05
    time.sleep(DT)