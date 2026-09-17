"""Open the NeuroMechFly model in MuJoCo's interactive viewer.

Nothing is driving the fly here -- this is for looking at the body: rotate with
the left mouse button, pan with the right, zoom with the scroll wheel, and use
the left panel to drag individual joints around.

    python scripts/01_view_model.py
    python scripts/01_view_model.py --terrain blocks
"""

import argparse

import _bootstrap  # noqa: F401
from flygym import launch_interactive_viewer

from flyplay.build import TERRAINS, build


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--terrain", default="flat", choices=sorted(TERRAINS), help="Scenario to load."
    )
    args = parser.parse_args()

    print(f"Building '{args.terrain}': {TERRAINS[args.terrain].description}")
    fs = build(args.terrain)
    print(
        f"model: {fs.sim.mj_model.nq} positions, {fs.sim.mj_model.nv} velocities, "
        f"{fs.sim.mj_model.nu} actuators, dt = {fs.sim.timestep} s"
    )
    print("Close the viewer window to exit.")
    launch_interactive_viewer(fs.sim.mj_model, fs.sim.mj_data)
    fs.close()


if __name__ == "__main__":
    main()
