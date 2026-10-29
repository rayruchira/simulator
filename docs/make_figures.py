# SPDX-License-Identifier: GPL-3.0-or-later
"""Makes the README's figures (docs/*.png).

    uv run --with matplotlib --with trimesh --with shapely --with networkx python docs/make_figures.py
"""
import os
import pathlib

os.environ.setdefault("MUJOCO_GL", "egl")
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle, Polygon, Rectangle  # noqa: E402

from climbing_sim import JOINTS, ClimbingSim  # noqa: E402
from climbing_sim import wall as W  # noqa: E402

OUT = pathlib.Path(__file__).parent
BLUE, ORANGE, GREY = "#1f77b4", "#d9622b", "#555555"


def servo_geoms(m, d) -> dict:
    """joint name -> the visual geom of the XL330 that drives it: the servo case whose output axis is the joint's
    axis (one case per joint)."""
    cases = [g for g in range(m.ngeom) if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
             and "xl330" in m.mesh(m.geom_dataid[g]).name]
    out = {}
    for n in JOINTS:
        j = m.joint(n).id
        v = d.geom_xpos[cases] - d.xanchor[j]
        along = v @ d.xaxis[j]
        off = np.linalg.norm(v - along[:, None] * d.xaxis[j], axis=1)
        out[n] = cases[int(np.argmin(off + 0.2 * np.abs(along)))]
    assert len(set(out.values())) == len(JOINTS), "one servo case per joint"
    return out


LIMB_COLOURS = {"head": "#7f7f7f", "right arm": "#d62728", "left arm": "#1f77b4", "right leg": "#ff7f0e",
                "left leg": "#2ca02c"}


def limb_of(name: str) -> str:
    if name == "head":
        return "head"
    side, part = name.split("_")[:2]
    return f"{side} {'arm' if part in ('shoulder', 'elbow') else 'leg'}"


def ortho_views(m, d, lookat, height_m, azimuths, size=(430, 700)):
    """For each azimuth: the colour image and the segmentation (geom id per pixel, -1 for none) of an
    orthographic view ``height_m`` metres high, centred on ``lookat``."""
    m.vis.global_.orthographic = 1
    m.vis.global_.fovy = height_m
    m.vis.global_.offwidth, m.vis.global_.offheight = size
    r = mujoco.Renderer(m, size[1], size[0])
    cam = mujoco.MjvCamera()
    cam.lookat[:] = lookat
    opt = mujoco.MjvOption()
    opt.sitegroup[:] = 0
    out = []
    for az in azimuths:
        cam.distance, cam.azimuth, cam.elevation = 1.5, az, 0.0
        r.disable_segmentation_rendering()
        r.update_scene(d, cam, opt)
        img = r.render()
        r.enable_segmentation_rendering()
        r.update_scene(d, cam, opt)
        seg = r.render()
        out.append((img, np.where(seg[..., 1] == mujoco.mjtObj.mjOBJ_GEOM, seg[..., 0], -1)))
    r.close()
    m.vis.global_.orthographic = 0
    return out


def robot_joints():
    """Each of the 19 servos, coloured by limb and numbered with its index in JOINTS, on the robot standing in its
    reset() pose, from four sides; the rest of the robot is see-through so the servos inside the trunk show."""
    sim = ClimbingSim("N2", cameras=[])
    sim.reset()
    m, d = sim.model, sim.data
    servo = servo_geoms(m, d)
    robot = [g for g in range(m.ngeom) if m.body_rootid[m.geom_bodyid[g]] == m.body("trunk").id
             and m.geom_group[g] != 3]
    visual = m.geom_rgba.copy()
    m.geom_rgba[:, 3] = 0.0                                     # the wall, holds, floor and collision shapes hidden
    for g in robot:
        m.geom_rgba[g] = [0.75, 0.75, 0.75, 0.22]
    for n, g in servo.items():
        m.geom_rgba[g] = list(matplotlib.colors.to_rgb(LIMB_COLOURS[limb_of(n)])) + [1.0]
    m.geom_matid[:] = -1
    views = [(90.0, "from behind"), (-90.0, "from the front (the wall side)"), (180.0, "from its right side"),
             (0.0, "from its left side")]
    lookat = np.array([d.qpos[0], d.qpos[1], 0.166])            # floor to head top
    imgs = ortho_views(m, d, lookat, 0.345, [az for az, _ in views])
    # where each servo is: what can be seen of it with the robot opaque (seg_out), and for the servos inside the
    # trunk, never seen from outside, where they are behind the see-through parts (seg_in: only the servos drawn)
    m.geom_rgba[robot, 3] = 1.0
    seg_out = [seg for _, seg in ortho_views(m, d, lookat, 0.345, [az for az, _ in views])]
    for g in robot:
        if g not in servo.values():
            m.geom_rgba[g, 3] = 0.0
    seg_in = [seg for _, seg in ortho_views(m, d, lookat, 0.345, [az for az, _ in views])]
    m.geom_rgba[:] = visual
    inside = [n for n in JOINTS if all(np.count_nonzero(sg == servo[n]) < 150 for sg in seg_out)]

    fig, axs = plt.subplots(1, 5, figsize=(19, 7.6), gridspec_kw=dict(width_ratios=[1, 1, 1, 1, 0.9]))
    shown = set()
    for ax, (img, _), so, si, (az, title) in zip(axs, imgs, seg_out, seg_in, views):
        ax.imshow(np.where(img.sum(-1, keepdims=True) == 0, 255, img))      # white background
        # the most visible servos first; a number that would cover another one is left out
        # a servo is numbered where it can be seen (or, inside the trunk, through it), at the middle of its case
        seen = sorted(((np.count_nonzero((si if n in inside else so) == servo[n]), i, n) for i, n in enumerate(JOINTS)),
                      reverse=True)
        placed = []
        for count, i, n in seen:
            ys, xs = np.nonzero(si == servo[n])
            if count < 150:                                     # (almost) hidden in this view
                continue
            x, y = np.median(xs), np.median(ys)
            if any(np.hypot(x - px, y - py) < 30 for px, py in placed):
                continue
            placed.append((x, y))
            shown.add(n)
            ax.text(x, y, str(i), ha="center", va="center", fontsize=10, weight="bold", color="black",
                    bbox=dict(boxstyle="circle,pad=0.25", fc="white", ec=LIMB_COLOURS[limb_of(n)], lw=2))
        ax.set_title(title, fontsize=11)
        ax.set_axis_off()
    assert shown == set(JOINTS), f"servos numbered in no view: {set(JOINTS) - shown}"
    axs[2].set_title("from its right side\n(back on the left, the wall on the right \u2192)", fontsize=11)
    axs[3].set_title("from its left side\n(\u2190 the wall on the left, back on the right)", fontsize=11)
    axs[0].set_title("from behind\n(its right on the right)", fontsize=11)
    axs[1].set_title("from the front, the wall side\n(its right on the left)", fontsize=11)

    leg = axs[-1]
    leg.set_axis_off()
    y = 0.98
    for limb, colour in LIMB_COLOURS.items():
        leg.text(0.02, y, limb, color=colour, weight="bold", fontsize=11, transform=leg.transAxes, va="top")
        y -= 0.045
        for i, n in enumerate(JOINTS):
            if limb_of(n) == limb:
                leg.text(0.06, y, f"{i:2d}  {n}", fontsize=10, family="monospace", transform=leg.transAxes,
                         va="top")
                y -= 0.04
        y -= 0.02
    fig.suptitle("The 19 servos: their index in JOINTS (the order of every joint array)", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "joints.png", dpi=100, bbox_inches="tight")
    plt.close(fig)


def start_poses():
    """reset() and reset_hanging(), side by side."""
    imgs = []
    for f, title in (("reset", "reset(): standing on the floor"), ("reset_hanging", "reset_hanging(): on the start holds")):
        sim = ClimbingSim("N2", cameras=[])
        getattr(sim, f)()
        r = mujoco.Renderer(sim.model, 600, 480)
        cam = mujoco.MjvCamera()
        cam.lookat[:] = sim.trunk_pose()[0] + [0, 0, 0.02]
        cam.distance, cam.azimuth, cam.elevation = 0.75, 125.0, -15.0
        r.update_scene(sim.data, cam)
        imgs.append((r.render(), title))
        r.close()
    fig, axs = plt.subplots(1, 2, figsize=(10, 6.2))
    for ax, (img, title) in zip(axs, imgs):
        ax.imshow(img)
        ax.set_title(title, fontsize=11)
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(OUT / "start_poses.png", dpi=100, bbox_inches="tight")
    plt.close(fig)


def arrow(ax, p, v, label, color, **kw):
    ax.annotate("", p + v, p, arrowprops=dict(arrowstyle="-|>", color=color, lw=2))
    off = v / np.linalg.norm(v) * 0.05
    ax.text(*(p + v + off * (2.2 if abs(v[0]) > abs(v[1]) else 0.8)), label, color=color, ha="center", va="center", fontsize=11, weight="bold", **kw)


def frames():
    """The world frame and the wall frame, side by side on the same drawing of the wall (oblique projection: x right,
    z up, the depth y drawn up and to the left), seen from the climber's side."""
    from climbing_sim import aruco
    k = 0.7 * np.array([np.cos(np.radians(145)), np.sin(np.radians(145))])
    Wd, H, T = W.WALL_WIDTH, W.WALL_HEIGHT, 0.03                # the board drawn 3 x thicker

    def P(x, y, z):                                             # world (m) -> drawing
        return np.array([x, z]) + y * k

    def wall(ax):
        def face(pts, **kw):
            ax.add_patch(Polygon([P(*q) for q in pts], **kw))
        face([(-0.3, -0.55, 0), (Wd + 0.3, -0.55, 0), (Wd + 0.3, 0.35, 0), (-0.3, 0.35, 0)], fc="#eef1f3",
             ec="none", zorder=0)
        face([(0, 0, 0), (0, T, 0), (0, T, H), (0, 0, H)], fc=WOOD_EDGE, ec="#7d6037", lw=1)       # left edge
        face([(0, 0, H), (Wd, 0, H), (Wd, T, H), (0, T, H)], fc="#cfae78", ec="#7d6037", lw=1)     # top edge
        face([(0, 0, 0), (Wd, 0, 0), (Wd, 0, H), (0, 0, H)], fc=WOOD, ec="#7d6037", lw=1.2)        # front face
        gx, gz = np.meshgrid(W.GRID_X0 + np.arange(W.GRID_COLS) * W.HOLE_PITCH,
                             W.GRID_Z0 + np.arange(W.GRID_ROWS) * W.HOLE_PITCH)
        ax.scatter(gx.ravel(), gz.ravel(), s=0.35, color="#a08458", lw=0, zorder=2)
        for mid, (x0, z0, x1, z1) in W.marker_zones().items():
            ax.imshow(aruco.image(mid, px_per_cell=20), cmap="gray", extent=(x0, x1, z0, z1), zorder=3,
                      interpolation="nearest")
            ax.add_patch(Rectangle((x0, z0), x1 - x0, z1 - z0, fc="none", ec="#9e9e9e", lw=0.6, zorder=3))
        for h in W.Wall.load("N2").holds:
            ax.add_patch(Polygon(h.outline(), fc=HOLD_RGB, ec="#b4532a", lw=0.4, zorder=4))

    def arrow(ax, v, name, colour, where, dashed=False):
        p0, p1 = P(0, 0, 0), P(*(0.34 * np.asarray(v, float)))
        from matplotlib import patheffects
        a = ax.annotate("", p1, p0, zorder=10, arrowprops=dict(arrowstyle="-|>", color=colour, lw=2.6,
                                                              mutation_scale=20, shrinkA=0, shrinkB=0,
                                                              ls=(0, (3, 2)) if dashed else "-"))
        a.arrow_patch.set_path_effects([patheffects.withStroke(linewidth=6, foreground="white")])
        ha, va, dx, dy = where
        ax.text(p1[0] + dx, p1[1] + dy, name, ha=ha, va=va, fontsize=11, color=colour, weight="bold", zorder=11,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.92))

    fig, axs = plt.subplots(1, 2, figsize=(14, 7.6))
    right_of, above = ("left", "center", 0.04, 0), ("center", "bottom", 0, 0.04)
    down_right = ("left", "top", 0.02, -0.02)
    for ax, title, axes in [
            (axs[0], "World frame (MuJoCo): everything in ClimbingSim",
             [([1, 0, 0], "x", right_of, False), ([0, 0, 1], "z (up)", above, False),
              ([0, 1, 0], "y (into the wall)", ("center", "bottom", 0, 0.035), False)]),
            (axs[1], "Wall frame: markers, camera extrinsics, wall editor",
             [([1, 0, 0], "x", right_of, False), ([0, 0, 1], "y (up)", above, False),
              ([0, -1, 0], "z (out of the wall)", down_right, False)])]:
        wall(ax)
        colour = BLUE if ax is axs[0] else ORANGE
        for v, name, where, dashed in axes:
            arrow(ax, v, name, colour, where, dashed)
        ax.plot(*P(0, 0, 0), "o", color=INK, ms=8, mec="white", mew=2, zorder=12)
        ax.text(*(P(0, 0, 0) + [0.03, -0.03]), "origin", ha="left", va="top", fontsize=10, color=INK, zorder=12)
        if ax is axs[0]:
            ax.text(*P(Wd + 0.25, -0.52, 0), "climber's side (y < 0)", ha="right", va="bottom", fontsize=10,
                    color=INK)
        ax.set_title(title, fontsize=12, color=colour, weight="bold")
        ax.set_aspect("equal")
        ax.set_xlim(-0.62, Wd + 0.36)
        ax.set_ylim(-0.42, H + 0.06)
        ax.set_axis_off()
    fig.suptitle("Same origin: the bottom-left corner of the board's front face, on the floor; same x axis",
                 fontsize=13, y=0.97)
    fig.text(0.5, 0.03, f"board {Wd * 1000:.0f} x {H * 1000:.0f} mm, {W.GRID_COLS} x {W.GRID_ROWS} holes every "
             f"{W.HOLE_PITCH * 1000:.0f} mm, the four ArUco markers and the holds of wall N2", ha="center",
             fontsize=10, color=INK)
    fig.subplots_adjust(wspace=0.02, top=0.9, bottom=0.06)
    fig.savefig(OUT / "frames.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


WOOD, WOOD_EDGE, HOLD_RGB, BALL_RGB, INK = "#e3c48f", "#a8844f", "#e8743b", "#9aa0a6", "#2b2b2b"


def dim(ax, p0, p1, text, offset=(0, 0), color=INK, **kw):
    """A dimension line between p0 and p1 with its text at the middle (+ offset, in data units)."""
    ax.annotate("", p1, p0, arrowprops=dict(arrowstyle="<|-|>", color=color, lw=1.2, shrinkA=0, shrinkB=0,
                                            mutation_scale=10))
    mid = (np.asarray(p0) + p1) / 2 + offset
    ax.text(*mid, text, color=color, ha="center", va="center", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.9), **kw)


def hold_section():
    """The hold cut by the vertical plane through its socket (x = 0), in the hold frame (mm): a list of shapely
    polygons in (y, z), y towards the wall."""
    import shapely
    import trimesh
    mesh = trimesh.load(pathlib.Path(W.__file__).parent / "assets" / "hold" / "hold_visual.stl")
    loops = mesh.section(plane_origin=[0, 0, 0], plane_normal=[1.0, 0, 0]).discrete    # closed 3D polylines
    area = None
    for loop in loops:
        poly = shapely.Polygon(loop[:, 1:] * 1000).buffer(0)
        area = poly if area is None else area.symmetric_difference(poly)
    return list(getattr(area, "geoms", [area]))


def hold():
    """A hold, from the climber's side (rendered) and cut through its socket, with its dimensions (mm)."""
    h = W.Hold(10, 20)
    so, sb, pitch, r_ball = W.SOCKET_OUT * 1000, W.SOCKET_BELOW * 1000, W.HOLE_PITCH * 1000, 12.5
    # front: the board's holes, the hold's outline on the board, the ball in the socket
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 5.4))
    a.add_patch(Rectangle((-60, -45), 120, 90, fc=WOOD, ec="none"))
    sx, sz = h.socket[0] * 1000, h.socket[2] * 1000
    for cl in range(h.col - 4, h.col + 7):
        for rw in range(h.row - 3, h.row + 3):
            x, z = W.hole_position(cl, rw)[[0, 2]] * 1000 - [sx, sz]
            a.add_patch(Circle((x, z), W.HOLE_DIAMETER * 500, fc="#7a5a2e", ec="none"))
    a.add_patch(Polygon(W.HOLD_OUTLINE * 1000, fc=HOLD_RGB, ec="#b4532a", lw=1.5, alpha=0.95))
    a.add_patch(Circle((0, 0), r_ball, fc=BALL_RGB, ec="#5f6368", lw=1.2))
    a.plot(0, 0, "+", color="white", ms=8, mew=1.5)
    holes = [W.hole_position(cl, rw)[[0, 2]] * 1000 - [sx, sz] for cl, rw in h.holes]
    for (x, z), name, ha in zip(holes, ["(col, row)", "(col + 2, row)"], ["right", "left"]):
        a.add_patch(Circle((x, z), 2.8, fc="#c9ccd1", ec="#4a4f55", lw=1.2))
        a.annotate(name, (x, z), (x + (-10 if ha == "right" else 10), z + 14), ha=ha, fontsize=10, color=INK,
                   arrowprops=dict(arrowstyle="-", color=INK, lw=0.8),
                   bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.9))
    (x0, z0), (x1, _) = holes
    dim(a, (x0, z0 + 6), (x1, z0 + 6), f"{2 * pitch:.0f} mm")
    a.plot([x1 + 4, x1 + 30], [z0, z0], color=INK, lw=0.8, ls=":")
    a.plot([r_ball + 1, x1 + 30], [0, 0], color=INK, lw=0.8, ls=":")
    dim(a, (x1 + 27, 0), (x1 + 27, z0), f"{sb:.0f} mm", offset=(8, 0))
    a.annotate("socket: the ball rests here", (0, -r_ball), (0, -30), ha="center", fontsize=10, color=INK,
               arrowprops=dict(arrowstyle="-", color=INK, lw=0.8),
               bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.9))
    a.text(-56, -41, f"holes every {pitch:.0f} mm", fontsize=9, color=INK)
    a.set_title("Hold(col, row), seen from the climber's side\n(bolted on two holes of a row)", fontsize=11)
    a.set_aspect("equal")
    a.set_xlim(-60, 60)
    a.set_ylim(-45, 45)
    a.set_axis_off()
    # side: the hold cut through its socket; the wall on the right, the climber on the left
    for poly in hold_section():
        for ring, fc in [(poly.exterior, HOLD_RGB)] + [(i, "white") for i in poly.interiors]:
            b.add_patch(Polygon(np.array(ring.coords), fc=fc, ec="#b4532a", lw=1.2))
    t = W.WALL_THICKNESS * 1000
    b.add_patch(Rectangle((so, -38), t, 76, fc=WOOD, ec=WOOD_EDGE, lw=1.2))
    # an M3 bolt (holes 3.2 mm): its head on the hold's back plate, the nut behind the wall (drawn although the bolts
    # are 15 mm on each side of this cut)
    steel, steel_edge = "#c3c7cc", "#5f6368"
    plate = 23.0                                                # the plate's climber-side face at the bolt height
    shaft_end = so + t + 2.4 + 2.0
    b.add_patch(Rectangle((plate, sb - 1.5), shaft_end - plate, 3.0, fc=steel, ec=steel_edge, lw=1, zorder=5))
    for x in np.arange(plate + 0.5, shaft_end - 0.3, 0.5):         # the thread
        b.plot([x, x + 0.5], [sb - 1.5, sb + 1.5], color=steel_edge, lw=0.6, zorder=6)
    b.add_patch(Rectangle((plate - 2.5, sb - 2.75), 2.5, 5.5, fc=steel, ec=steel_edge, lw=1, zorder=6))   # head
    b.add_patch(Rectangle((so + t, sb - 2.75), 2.4, 5.5, fc=steel, ec=steel_edge, lw=1, zorder=6))       # nut
    b.annotate("M3 bolt and nut\n(one 15 mm each side of this cut)", (so + t + 2.4, sb + 2.75), (so + t + 4, sb + 14),
               fontsize=9, color=INK, arrowprops=dict(arrowstyle="-", color=INK, lw=0.8))
    b.add_patch(Circle((0, 0), r_ball, fc=BALL_RGB, ec="#5f6368", lw=1.2, zorder=3))
    b.plot(0, 0, "+", color="white", ms=8, mew=1.5, zorder=4)
    dim(b, (0, 24), (so, 24), f"{so:.0f} mm", offset=(0, 3))
    b.plot([0, 0], [r_ball + 1, 26], color=INK, lw=0.8, ls=":")
    dim(b, (-20, 0), (-20, sb), f"{sb:.0f} mm", offset=(-6, 0))
    b.plot([-22, -r_ball - 1], [0, 0], color=INK, lw=0.8, ls=":")
    b.plot([-22, plate - 3], [sb, sb], color=INK, lw=0.8, ls=":")
    b.annotate("ball, 25 mm diameter", (-r_ball * 0.7, -r_ball * 0.7), (-34, -30), fontsize=10, color=INK,
               arrowprops=dict(arrowstyle="-", color=INK, lw=0.8))
    b.text(so + t / 2, -34, "wall", ha="center", fontsize=10, color=INK, rotation=90)
    b.text(-40, 32, "\u2190 climber's side", fontsize=10, color=INK)
    b.set_title("Cut through the socket, seen from the side\n(the wall on the right)", fontsize=11)
    b.set_aspect("equal")
    b.set_xlim(-42, so + t + 22)
    b.set_ylim(-38, 38)
    b.set_axis_off()
    fig.tight_layout()
    fig.savefig(OUT / "hold.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


def control_loop():
    """What one sim.step(target) does: the loop (your code -> servos -> physics -> observation) and its timing."""
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    fig = plt.figure(figsize=(13, 8.6))
    ax = fig.add_axes([0, 0.42, 1, 0.58])
    ax.set_xlim(0, 13)
    ax.set_ylim(-0.55, 4.15)
    ax.set_axis_off()

    def box(x, y, w, h, title, text, fc, ec):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.18", fc=fc, ec=ec, lw=1.6))
        ax.text(x + w / 2, y + h - 0.28, title, ha="center", va="top", fontsize=12, weight="bold", color=INK)
        ax.text(x + w / 2, y + h - 0.72, text, ha="center", va="top", fontsize=10, color=INK, linespacing=1.5)

    def link(p0, p1, text="", rad=0.0, colour=INK, side=(0, 0.18)):
        ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=18, lw=1.8, color=colour,
                                     connectionstyle=f"arc3,rad={rad}"))
        if text:
            mid = (np.asarray(p0) + p1) / 2 + side
            ax.text(*mid, text, ha="center", va="center", fontsize=10, color=colour,
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none"))

    box(0.3, 1.3, 2.6, 2.0, "your code", "planner, policy...\nreads the observation,\ncomputes the targets",
        "#e8f0fb", BLUE)
    ax.add_patch(FancyBboxPatch((3.9, 0.55), 8.8, 3.55, boxstyle="round,pad=0.02,rounding_size=0.2", fc="#f7f7f7",
                                ec="#9e9e9e", lw=1.2, ls=(0, (4, 3))))
    ax.text(4.1, 3.95, "obs = sim.step(target): 20 ms of simulated time", ha="left", va="top", fontsize=12, weight="bold",
            family="monospace", color=INK)
    box(4.3, 1.3, 2.4, 2.0, "latency", "the target reaches\nthe servos 10 ms\nafter it is sent", "#fff4e5",
        "#e8a33b")
    box(7.2, 1.3, 2.4, 2.0, "19 servos", "firmware P gain kp_fw\nmotor model (BAM):\ntorque, current, friction",
        "#fde9e1", HOLD_RGB)
    box(10.1, 1.3, 2.4, 2.0, "physics", "MuJoCo, 10 steps\nof 2 ms: contacts,\nballs in the holds",
        "#e9f5ec", "#2ca02c")
    link((2.9, 2.6), (4.3, 2.6), "target\n19 angles (rad)", side=(0, 0.42))
    link((6.7, 2.3), (7.2, 2.3))
    link((9.6, 2.3), (10.1, 2.3))
    ax.add_patch(FancyArrowPatch((11.3, 1.3), (1.6, 1.3), arrowstyle="-|>", mutation_scale=18, lw=1.8, color=BLUE,
                                 connectionstyle="arc3,rad=-0.25"))
    ax.text(6.5, 0.12, "observation: what the real robot measures, with its delays\nq, dq (joints) · gyro, accel, "
            "gravity, quat (IMU) · current (motors) · time", ha="center", va="center", fontsize=10, color=BLUE,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=BLUE, lw=1))

    # timing, in simulated time: what your code sends, what the servos follow, the physics steps
    t = fig.add_axes([0.0, 0.0, 1.0, 0.40])
    t.set_xlim(-30, 75)
    t.set_ylim(-3.2, 4.0)
    t.set_axis_off()
    t.text(-29, 3.95, "Timing of three calls to step(), in simulated time", fontsize=12, weight="bold", color=INK,
           va="top")
    rows = {"your code sends": 1.9, "the servos follow": 0.9, "physics (MuJoCo)": -0.1}
    for name, y in rows.items():
        t.text(-2, y + 0.35, name, ha="right", va="center", fontsize=10, color=INK)
    for k, x0 in enumerate((0, 20, 40)):
        name = ["k", "k + 1", "k + 2"][k]
        t.add_patch(Rectangle((x0 + 0.3, 1.9), 19.4, 0.7, fc="#e8f0fb", ec=BLUE, lw=1))
        t.text(x0 + 10, 2.25, f"step(target {name})", ha="center", va="center", fontsize=9, family="monospace",
               color=INK)
        prev = ["k - 1", "k", "k + 1"][k]                  # the first 10 ms: the last target, then this one
        t.add_patch(Rectangle((x0 + 0.3, 0.9), 9.4, 0.7, fc="#fde9e1", ec=HOLD_RGB, lw=1))
        t.text(x0 + 5, 1.25, f"target {prev}", ha="center", va="center", fontsize=8, color=INK)
        t.add_patch(Rectangle((x0 + 10.3, 0.9), 9.4, 0.7, fc="#fde9e1", ec=HOLD_RGB, lw=1))
        t.text(x0 + 15, 1.25, f"target {name}", ha="center", va="center", fontsize=8, color=INK)
        for i in range(10):
            t.add_patch(Rectangle((x0 + 2 * i + 0.15, -0.1), 1.7, 0.7, fc="#e9f5ec", ec="#2ca02c", lw=0.8))
    t.annotate("", (10.3, 1.6), (0.3, 1.9), arrowprops=dict(arrowstyle="-|>", color="#e8a33b", lw=1.8,
                                                           mutation_scale=14))
    t.text(5, 1.75, "10 ms latency", ha="center", va="center", fontsize=8, color="#b26b00",
           bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none"))
    t.annotate("one physics step: 2 ms", (1.0, -0.1), (1.0, -0.75), ha="left", va="top", fontsize=9, color=INK,
               arrowprops=dict(arrowstyle="-", color=INK, lw=0.8))
    for x in (0, 20, 40, 60):
        t.plot([x, x], [-0.25, 2.75], color="#bdbdbd", lw=0.8, ls=":", zorder=0)
        t.text(x, 2.85, f"{x} ms", ha="center", va="bottom", fontsize=9, color=GREY)
    t.text(-29, -1.55, "These are simulated milliseconds. step() returns as soon as MuJoCo has computed its 20 ms: "
           "nothing waits for the clock.\nOn a laptop that takes about 2 ms of real time without images (the "
           "simulation runs faster than real time), more with images.\nOn the real robot the same loop runs in "
           "real time, at 50 Hz.", ha="left", va="top", fontsize=10, color=INK, linespacing=1.6,
           bbox=dict(boxstyle="round,pad=0.5", fc="#fff8e1", ec="#e8a33b", lw=1))
    fig.savefig(OUT / "control_loop.png", dpi=100, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    robot_joints()
    start_poses()
    frames()
    hold()
    control_loop()
    print("written:", sorted(p.name for p in OUT.glob("*.png")))
