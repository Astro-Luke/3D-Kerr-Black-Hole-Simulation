import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Patch

# ============================================================
# Kerr black hole visualization (Boyer-Lindquist-style)
# Modes:
#   - gif         : left 2D section + right 3D animation
#   - interactive : live 3D animation, rotatable with mouse
#
# Units: G = c = M = 1
# ============================================================

# ------------------------------------------------------------
# Global parameters
# ------------------------------------------------------------
L3D = 2.7
L2D = 2.7

fps = 16
n_frames = 140

# Mesh for 3D surfaces
theta_3d = np.linspace(0, np.pi, 140)
phi_3d = np.linspace(0, 2*np.pi, 180)
TH3, PH3 = np.meshgrid(theta_3d, phi_3d)

# Angle for 2D meridional section (full closed curve)
theta_2d = np.linspace(0, 2*np.pi, 900)

# Colors
COLOR_INNER = "#c43c39"   # red
COLOR_OUTER = "#2c6bed"   # blue
COLOR_ERGO  = "#bdbdbd"   # gray

ALPHA_INNER = 0.68
ALPHA_OUTER = 0.42
ALPHA_ERGO  = 0.24

# Fixed camera for initial 3D view
ELEV = 20
AZIM = 42


# ------------------------------------------------------------
# Kerr radii
# ------------------------------------------------------------
def r_plus(a):
    return 1.0 + np.sqrt(np.maximum(0.0, 1.0 - a**2))

def r_minus(a):
    return 1.0 - np.sqrt(np.maximum(0.0, 1.0 - a**2))

def r_ergosphere_outer(a, th):
    return 1.0 + np.sqrt(np.maximum(0.0, 1.0 - a**2 * np.cos(th)**2))


# ------------------------------------------------------------
# Periodic spin parameter: 0 -> 1 -> 0 without discontinuity
# ------------------------------------------------------------
def spin_parameter(frame, n_frames):
    phase = 2.0 * np.pi * frame / n_frames
    return 0.5 * (1.0 - np.cos(phase))


# ------------------------------------------------------------
# Boyer-Lindquist-style coordinates
# ------------------------------------------------------------
def bl_cartesian_3d(r, th, ph):
    x = r * np.sin(th) * np.cos(ph)
    y = r * np.sin(th) * np.sin(ph)
    z = r * np.cos(th)
    return x, y, z

def bl_meridional_section_full(r, th):
    """
    Full closed curve in the x-z plane.
    Using theta from 0 to 2pi gives both right and left sides.
    """
    x = r * np.sin(th)
    z = r * np.cos(th)
    return x, z

def outer_horizon_surface(a):
    r = r_plus(a)
    return bl_cartesian_3d(r, TH3, PH3)

def inner_horizon_surface(a):
    r = r_minus(a)
    return bl_cartesian_3d(r, TH3, PH3)

def ergosphere_surface(a):
    r = r_ergosphere_outer(a, TH3)
    return bl_cartesian_3d(r, TH3, PH3)

def outer_horizon_curve_2d(a):
    th = theta_2d
    r = r_plus(a) * np.ones_like(th)
    return bl_meridional_section_full(r, th)

def inner_horizon_curve_2d(a):
    th = theta_2d
    r = r_minus(a) * np.ones_like(th)
    return bl_meridional_section_full(r, th)

def ergosphere_curve_2d(a):
    th = theta_2d
    # in the x-z meridional plane, polar angle is theta itself
    r = r_ergosphere_outer(a, th)
    return bl_meridional_section_full(r, th)


# ------------------------------------------------------------
# Styling helpers
# ------------------------------------------------------------
def style_3d_axes(ax):
    ax.set_xlim(-L3D, L3D)
    ax.set_ylim(-L3D, L3D)
    ax.set_zlim(-L3D, L3D)
    ax.set_box_aspect([1, 1, 1])

    ticks = np.arange(-2, 3, 1)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_zticks(ticks)

    ax.set_xlabel("x", labelpad=8)
    ax.set_ylabel("y", labelpad=8)
    ax.set_zlabel("z", labelpad=8)

    ax.grid(True)
    ax.xaxis.pane.set_alpha(0.06)
    ax.yaxis.pane.set_alpha(0.06)
    ax.zaxis.pane.set_alpha(0.06)

    ax.plot([-L3D, L3D], [0, 0], [0, 0], color="black", lw=0.8)
    ax.plot([0, 0], [-L3D, L3D], [0, 0], color="black", lw=0.8)
    ax.plot([0, 0], [0, 0], [-L3D, L3D], color="black", lw=0.8)

def style_2d_axes(ax):
    ax.set_xlim(-L2D, L2D)
    ax.set_ylim(-L2D, L2D)
    ax.set_aspect("equal", adjustable="box")

    ticks = np.arange(-2, 3, 1)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

    ax.set_xlabel("x")
    ax.set_ylabel("z")

    ax.grid(True, alpha=0.35)

    ax.axhline(0, color="black", lw=0.8)
    ax.axvline(0, color="black", lw=0.8)


# ------------------------------------------------------------
# 3D plotting for a single frame
# ------------------------------------------------------------
def plot_3d_surfaces(ax, a):
    Xe, Ye, Ze = ergosphere_surface(a)
    Xo, Yo, Zo = outer_horizon_surface(a)

    # Ergosphere with light mesh
    ax.plot_surface(
        Xe, Ye, Ze,
        rstride=4, cstride=4,
        color=COLOR_ERGO,
        alpha=ALPHA_ERGO,
        linewidth=0.30,
        edgecolor=(0.45, 0.45, 0.45, 0.32),
        shade=True,
        antialiased=True
    )

    # Outer horizon smooth
    ax.plot_surface(
        Xo, Yo, Zo,
        rstride=2, cstride=2,
        color=COLOR_OUTER,
        alpha=ALPHA_OUTER,
        linewidth=0,
        edgecolor='none',
        shade=True,
        antialiased=True
    )

    rm = r_minus(a)
    if rm > 1e-8:
        Xi, Yi, Zi = inner_horizon_surface(a)
        ax.plot_surface(
            Xi, Yi, Zi,
            rstride=2, cstride=2,
            color=COLOR_INNER,
            alpha=ALPHA_INNER,
            linewidth=0,
            edgecolor='none',
            shade=True,
            antialiased=True
        )

    style_3d_axes(ax)


# ------------------------------------------------------------
# 2D plotting for a single frame
# ------------------------------------------------------------
def plot_2d_section(ax, a):
    xe, ze = ergosphere_curve_2d(a)
    xo, zo = outer_horizon_curve_2d(a)
    xi, zi = inner_horizon_curve_2d(a)

    # Fill order: ergo -> outer -> inner
    ax.fill(xe, ze, color=COLOR_ERGO, alpha=ALPHA_ERGO, lw=1.0, ec=COLOR_ERGO)
    ax.fill(xo, zo, color=COLOR_OUTER, alpha=ALPHA_OUTER, lw=1.0, ec=COLOR_OUTER)

    if r_minus(a) > 1e-8:
        ax.fill(xi, zi, color=COLOR_INNER, alpha=ALPHA_INNER, lw=1.0, ec=COLOR_INNER)

    # Boundaries
    ax.plot(xe, ze, color="#8c8c8c", lw=1.4, label="Ergosphere")
    ax.plot(xo, zo, color=COLOR_OUTER, lw=1.6, label="Outer horizon")

    if r_minus(a) > 1e-8:
        ax.plot(xi, zi, color=COLOR_INNER, lw=1.6, label="Inner horizon")

    # Poles where ergosphere meets outer horizon
    rp = r_plus(a)
    ax.scatter([0, 0], [rp, -rp], s=26, color="black", zorder=6)

    style_2d_axes(ax)
    ax.set_title("2D meridional section", fontsize=12)


# ------------------------------------------------------------
# Interactive animated view
# ------------------------------------------------------------
def show_interactive():
    fig = plt.figure(figsize=(11.5, 8))
    ax = fig.add_subplot(111, projection='3d')
    fig.subplots_adjust(right=0.80)

    legend_handles = [
        Patch(facecolor=COLOR_INNER, edgecolor='black', linewidth=0.5,
              alpha=ALPHA_INNER, label='Inner horizon'),
        Patch(facecolor=COLOR_OUTER, edgecolor='black', linewidth=0.5,
              alpha=ALPHA_OUTER, label='Outer horizon'),
        Patch(facecolor=COLOR_ERGO, edgecolor='black', linewidth=0.5,
              alpha=ALPHA_ERGO, label='Ergosphere')
    ]

    # View state: keep whatever the user sets with the mouse
    view_state = {"elev": ELEV, "azim": AZIM, "initialized": False}

    def update(frame):
        # Save current user view before clearing
        if view_state["initialized"]:
            view_state["elev"] = ax.elev
            view_state["azim"] = ax.azim

        ax.cla()

        a = spin_parameter(frame, n_frames)
        plot_3d_surfaces(ax, a)

        # Restore current view so user mouse rotation is preserved frame to frame
        ax.view_init(elev=view_state["elev"], azim=view_state["azim"])
        view_state["initialized"] = True

        ax.set_title(f"Kerr black hole in 3D   |   a = {a:.3f}", pad=18, fontsize=14)
        ax.text2D(0.02, 0.96, "Coordinates: Boyer-Lindquist-style visualization",
                  transform=ax.transAxes, fontsize=10)
        ax.text2D(0.02, 0.92, "Mouse: drag to rotate while animation runs",
                  transform=ax.transAxes, fontsize=10)

        ax.legend(
            handles=legend_handles,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=True,
            borderpad=0.8
        )

    ani = FuncAnimation(
        fig,
        update,
        frames=n_frames,
        interval=1000/fps,
        repeat=True,
        blit=False
    )

    plt.show()


# ------------------------------------------------------------
# GIF mode: left 2D + right 3D
# ------------------------------------------------------------
def make_gif():
    fig = plt.figure(figsize=(14, 7.6))
    ax2d = fig.add_subplot(1, 2, 1)
    ax3d = fig.add_subplot(1, 2, 2, projection='3d')
    fig.subplots_adjust(wspace=0.15, right=0.86)

    legend_handles = [
        Patch(facecolor=COLOR_INNER, edgecolor='black', linewidth=0.5,
              alpha=ALPHA_INNER, label='Inner horizon'),
        Patch(facecolor=COLOR_OUTER, edgecolor='black', linewidth=0.5,
              alpha=ALPHA_OUTER, label='Outer horizon'),
        Patch(facecolor=COLOR_ERGO, edgecolor='black', linewidth=0.5,
              alpha=ALPHA_ERGO, label='Ergosphere')
    ]

    def update(frame):
        ax2d.cla()
        ax3d.cla()

        a = spin_parameter(frame, n_frames)

        plot_2d_section(ax2d, a)
        plot_3d_surfaces(ax3d, a)
        ax3d.view_init(elev=ELEV, azim=AZIM)

        fig.suptitle(f"Kerr black hole   |   a = {a:.3f}", fontsize=16, y=0.97)

        ax2d.text(
            0.03, 0.96,
            "At the poles: ergosphere meets outer horizon",
            transform=ax2d.transAxes,
            fontsize=10,
            va="top"
        )

        ax3d.text2D(
            0.02, 0.96,
            "3D view",
            transform=ax3d.transAxes,
            fontsize=10
        )

        ax3d.legend(
            handles=legend_handles,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=True,
            borderpad=0.8
        )

    ani = FuncAnimation(fig, update, frames=n_frames, interval=1000/fps)

    outname = "kerr_2d_3d_loop.gif"
    ani.save(outname, writer=PillowWriter(fps=fps))
    plt.close(fig)

    print(f"GIF salvata come: {outname}")


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    mode = input("Scegli modalità ('gif' oppure 'interactive'): ").strip().lower()

    if mode == "gif":
        make_gif()
    elif mode == "interactive":
        show_interactive()
    else:
        print("Modalità non riconosciuta. Usa 'gif' oppure 'interactive'.")

if __name__ == "__main__":
    main()