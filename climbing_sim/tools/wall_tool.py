# SPDX-License-Identifier: GPL-3.0-or-later
"""Make or check a wall (a set of holds) for the simulator.

    python -m climbing_sim.tools.wall_tool new my_wall.json 10,5 14,5 ...   # holds by their LEFT hole (col,row)
    python -m climbing_sim.tools.wall_tool random my_wall.json --level 3 --seed 7   # route of N1-N4 (seed 0: N1-N3)
    python -m climbing_sim.tools.wall_tool check my_wall.json               # overlaps, holds on a marker, blocked ways out
    python -m climbing_sim.tools.wall_tool show N2                          # the robot on it, in MuJoCo

A hold uses holes (col, row) and (col + 2, row); columns 0-51 from the left (seen from the climbing side),
rows 0-92 from the bottom.  Optional start holds: --start left_hand=18,23 right_hand=29,23 left_foot=21,2
right_foot=26,2 (they are added to the holds).
"""
import argparse
import sys

from ..wall import Hold, Wall


def _cell(s):
    """(col, row) of a "col,row" argument."""
    c, r = s.split(",")
    return int(c), int(r)


def main(argv=None):
    """Run the command line (``argv``: the arguments, sys.argv[1:] by default)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    n = sub.add_parser("new"); n.add_argument("path"); n.add_argument("holds", nargs="*")
    n.add_argument("--start", nargs="*", default=[]); n.add_argument("--name", default=None)
    r = sub.add_parser("random"); r.add_argument("path"); r.add_argument("--level", type=int, default=2, choices=[1, 2, 3, 4])
    r.add_argument("--seed", type=int, default=0)
    c = sub.add_parser("check"); c.add_argument("wall")
    s = sub.add_parser("show"); s.add_argument("wall")
    a = ap.parse_args(argv)
    if a.cmd == "new":
        w = Wall(name=a.name or a.path.rsplit("/", 1)[-1].rsplit(".", 1)[0])
        for kv in a.start:
            limb, cell = kv.split("=")
            w.start[limb] = w.add(*_cell(cell))
        for h in a.holds:
            if Hold(*_cell(h)) not in w.holds:
                w.add(*_cell(h))
        w.to_json(a.path)
        print(f"{a.path}: {len(w.holds)} holds")
    elif a.cmd == "random":
        from ..routes import generate_wall
        w = generate_wall(a.level, a.seed)
        w.to_json(a.path)
        print(f"{a.path}: {w.name}, {len(w.holds)} holds")
    elif a.cmd == "check":
        w = Wall.load(a.wall)
        bad = [(h, w.clashes(h)) for h in w.holds if w.clashes(h)]
        print(f"{w.name}: {len(w.holds)} holds, start {', '.join(f'{l}=({h.col},{h.row})' for l, h in w.start.items()) or 'none'}")
        for h, cl in bad:
            print(f"  OVERLAP: {h} with {cl}")
        for h in w.holds:
            if w.exit_blocks(h):
                print(f"  way out blocked: {h} and {w.exit_blocks(h)}")
        on_markers = [(h, h.on_markers()) for h in w.holds if h.on_markers()]
        for h, m in on_markers:
            print(f"  ON A CORNER MARKER: {h} covers marker(s) {m}")
        print("ok" if not bad and not on_markers else f"{len(bad)} clash(es), {len(on_markers)} on a marker")
    elif a.cmd == "show":
        from .viewer import show
        show(a.wall)


if __name__ == "__main__":
    sys.exit(main())
