# SPDX-License-Identifier: GPL-3.0-or-later
"""Render the two external cameras (set with climbing-sim-cameras) and the overview to PNG files.

    python examples/cameras.py [cameras.json]
"""
import sys

from PIL import Image

from climbing_sim import ClimbingSim

sim = ClimbingSim("N2", cameras=sys.argv[1] if len(sys.argv) > 1 else None)
sim.reset_hanging()
for name in [*sim.cameras, "overview"]:
    Image.fromarray(sim.render(name)).save(f"{name}.png")
    print(f"{name}.png")
sim.close()                                            # free the renderers before Python exits
