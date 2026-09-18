# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A workbench built on **FlyGym 2.1.0 / NeuroMechFly v2** (EPFL Ramdya Lab) — a
whole-body *Drosophila* biomechanics simulator on MuJoCo. Two goals drive every
design decision here: **watch the fly move in real time**, and **see what it
learns** under different conditions. Legibility beats raw training throughput.

Not a package. No `pyproject.toml` and no test framework; `requirements.txt`
pins what the scripts import (`requirements-gpu.txt` adds MuJoCo Warp).
`flyplay/` is a plain directory importable because every script does
`import _bootstrap` first.

A public git repository, https://github.com/Daejun/flyplay. `.gitignore` leaves
out both virtual environments, `out/` and the two FlyGym reference clones
(`flygym-src/`, `flygym-v1-src/`); commits use the account's noreply address.
Commit or push only when the user asks.

**Branches and releases.** Work happens on `main`. `release` is the GitHub default
branch, the one people clone, and moves only to a commit that installed and ran
from a fresh clone: a new venv from `requirements.txt` (pinned to the versions
tested here), `13_mb_check.py --no-plot` and `10_sense_check.py` passing, the
sandbox viewer answering `/state`. Each such commit gets a tag `vX.Y.Z` and a
GitHub release made with `gh` (`C:\Program Files\GitHub CLI\gh.exe`, not on the
Git Bash PATH). Keep the README's quick start in step with what that check ran.
Docs use paths relative to the project root, never this machine's home folder.

## Running things

Always use the venv interpreter explicitly — there is no activation step in
these workflows:

```bash
.venv\Scripts\python.exe scripts\09_web_viewer.py --odor --pillars 4
```

`.venv` is FlyGym **2.x** (Python 3.12, torch CPU-only). `.venv-v1` is a
separate FlyGym **1.x** install — the two APIs are incompatible. 1.x is only
for consulting the odour-plume, connectome-vision and path-integration
tutorials that were never ported; nothing in `flyplay/` imports it.

Scripts are numbered in dependency order and each carries a usage block in its
module docstring. The ones that matter most:

```bash
.venv\Scripts\python.exe scripts\10_sense_check.py
```

`10_sense_check.py` is the closest thing to a test suite: 10 pass/fail checks
that the odour and vision wiring produce correctly-signed, monotonic signals,
and that the eyes read painted floor colour on the correct side.
**Run it after touching `flyplay/odor.py`, `flyplay/vision.py`,
`flyplay/arena.py`, `flyplay/visual_pathway.py`, or sensor placement in
`build.py`.** If it fails, training cannot succeed — fix it before doing
anything else.

```bash
.venv\Scripts\python.exe scripts\13_mb_check.py
```

The mushroom-body counterpart: 38 pass/fail checks (16 smell, 13 vision, 9 on
the sandbox's connectome front end), pure numpy, seconds. **Run it after touching
`flyplay/olfactory.py`, `flyplay/visual_pathway.py`, `flyplay/mushroom_body.py`,
`flyplay/data/`, or the omission feedback in `flyplay/conditioning.py`.** The behavioural experiments are
`14_run_conditioning.py --experiment compartments|extinction` (~8 min each on 16
cores) and `--experiment colour|brightness --seeds 10` (~2 h each: eyes rendered
in every fly), each with its own output dir, then `15_analyze_mb.py --dir <that
dir>`. Trained flies are filed and restored with `16_memory.py`. The planned
programme these belong to, and its progress, is in [PLAN.md](PLAN.md).

```bash
.venv\Scripts\python.exe scripts\11_train_forage.py --resume --steps 150000 250000 250000 --n-envs 6 --checkpoint-every 8000
```

Three-stage curriculum with weight transfer. `--resume` continues each stage
from its newest checkpoint and skips finished ones; without it, the run
**deletes that stage's existing checkpoints** and starts over. Pass the same
`--steps` as the original run or the targets will not match.

```bash
.venv\Scripts\tensorboard.exe --logdir out/rl/tb
```

```bash
.venv\Scripts\python.exe scripts\17_run_experiment.py --set shock_odour --flies 8
```

Runs one of the sandbox's A/B experiment sets (`--list` names them) on all but
two cores, writing `out/experiments/<stamp>-<set>/`: every trial as it finishes,
replay files for the first flies, and at the end `trials.csv` and a Korean
inquiry report (`report.html`). The viewer's A/B lab starts the same script
with `--dir` and reads the same directory. Runs are the user's to start: they
fill the viewer's experiment list and take all but two cores.

```bash
.venv\Scripts\python.exe scripts\18_build_brain_data.py --download
```

Fetches DoOR's receptor responses and the hemibrain connectome into `data/raw/`
(git-ignored, 46 MB) and rebuilds the small tables in `flyplay/data/` that the
sandbox's olfactory front end reads. The tables are committed, so this is only
for changing what is extracted. `19_eye_azimuths.py` re-measures, by rendering,
which way each ommatidium looks (`flyplay/data/ommatidia_azimuth.npz`, read by
the central complex). The tables keep their sources' licences
(`flyplay/data/SOURCES.md`; DoOR's is CC BY-SA).

The sandbox viewer keeps its fly across restarts in a *session*,
`out/sandbox/sessions/<name>/` (`--session`, default `main`; `--fresh` starts a
new fly and sets the old state aside). A second viewer on the same session is
refused, so a test viewer on another port needs `--session <other name>`.

See [STATUS.md](STATUS.md) for what is currently trained and how to pick it up.

## Architecture

Three layers, each usable on its own. Read them in this order.

**`flyplay/build.py` — scene construction.** `TERRAINS` maps a name
(`flat` / `gapped` / `blocks` / `mixed`) to a `TerrainSpec` holding the world
factory, spawn height, bounds, and contact preset. `build()` returns a `FlySim`
wrapping the compiled `Simulation` plus cached MuJoCo index lookups
(`thorax_pos`, `heading`, `yaw_rate`, `leg_contacts`, `touching_pillar`, …).
Everything downstream goes through `FlySim` rather than touching `mj_data`.

**`flyplay/control.py` — walking.** `Walker` drives FlyGym's hybrid CPG turning
controller. The policy never commands joints; it emits a 2-element
**descending signal** that scales left/right tripod amplitude, mirroring
brain→VNC descending neurons. `RealtimePacer` throttles a loop to wall-clock
speed for live viewing.

**Environments — two families, same action space.**

| | `flyplay/env.py` `FlyNavEnv` | `flyplay/env_multimodal.py` `ForageEnv` |
|---|---|---|
| obs | 34-dim, **includes goal bearing** | 40-dim, **sensory only** |
| task | steer to a known goal | smell out food, dodge pillars by sight |
| scripts | 05–07 | 10–12 |

`ForageEnv` is the real target (NeuroMechFly v2 paper Fig. 5). Its point is
that neither sense substitutes for the other: odour is radial so it says
*where* but not *how*; pillars are visible but the food has no visual marker at
all. A `FlyNavEnv` policy is a valid initialisation for `ForageEnv` only
because the action space is identical — the observation spaces are not.

Supporting modules: `odor.py` (olfaction ported onto 2.x, which ships none),
`vision.py` (721 ommatidia × 2 → 6 analytic bilateral features), `arena.py`
(pillars and markers).

**A third stack: the mushroom body** (`olfactory.py` and `visual_pathway.py` →
`mushroom_body.py` → `conditioning.py`, scripts 13–16). This one does *not* use
PPO, and the difference is structural rather than stylistic. PPO learns across
250k steps between episodes; a mushroom body learns within a session, over
about a dozen trials, by local synaptic depression — so its weights must
survive what a `gym.Env` reset would destroy. `conditioning.py` is therefore an
experiment runner, not an environment. Do not try to wrap it in Gymnasium.

`scripts/09_web_viewer.py` is the main deliverable for the "watch it learn"
goal: MJPEG stream + neural dashboard in a browser, with `--follow <run>`
hot-swapping checkpoints while training runs.

## Invariants that are not obvious from the code

Each of these was found the expensive way. Violating them usually produces
**silently wrong behaviour, not an exception**.

**Props go in before `world.add_fly()`.** Every fly geom is `contype=0,
conaffinity=0`; collision comes only from explicit `<pair>` elements generated
at `add_fly()` time against `world.ground_geoms`. A pillar appended afterwards
is intangible.

**Geom groups 1 and 2 are invisible to the fly.** The eye renderer disables
them. That is how the odour marker shows in video but not in the ommatidia —
put visualisation-only props in group 2, solid obstacles in group 0.

**Renderers must be created on the thread that renders with them.** Hence
`FlyServer.build()` runs on the sim thread, not in `__init__`. With the env
(and its eye cameras) built on the main thread and the viewer's renderer on the
sim thread, mean ommatidia brightness dropped from 0.602 to **0.012** — the fly
goes effectively blind with no error raised. Both on the sim thread is fine,
including at the viewer's 1280×720 (checked: eye mosaic mean 0.43). After any
change to how renderers are created, look at the eye panel, not just for errors.

**Calibrate the visual horizon on an empty scene.** The floor already darkens
~337/721 ommatidia, so features use only the upper field. Calibrating with a
pillar in view pushes the horizon above it and obstacles read as zero forever.
`ForageEnv` parks pillars at `PARKING_SPOT` for this one reading.

**The walker's fast path must stay bit-identical to FlyGym's.** `Walker` runs
`FastTurningController` (FlyGym's hybrid turning controller with its output
order, phase-gain breakpoints and correction vectors built once),
`ContactForces` (FlyGym's `get_bodysegment_contact_forces` with its lookups
built once), `Walker.advance`, which steps physics as one `mj_step(nstep)`
call between controller updates, and one `PPoly` call for all six legs'
splines with the rest of the per-leg work in Python floats (88.7 us a call to
51.3; checked against the version it replaced over 5000 observations recorded
from a real run, joint angles, adhesion, both correction arrays and the CPG
phases equal on every one). Same arithmetic, same order: checked against
the stock pipeline (`from_sim`, `HybridTurningController`,
`apply_locomotion_action`, single `sim.step`s) over 6000 updates in a walled
room with stops and turns -- 0 mismatches in observations, ctrl, qpos, qvel and
controller state -- and by the sandbox fingerprint. A sandbox fly went from 1378
to 1212 ms per simulated second (0.73x to 0.83x). After a FlyGym upgrade, repeat
that comparison before trusting any result.

**Descending signal lives in [0.4, 1.6].** Below ~0.3 a tripod stalls, the fly
pivots about the stalled side, and **the turn direction flips**. `SIGNAL_LOW` /
`SIGNAL_HIGH` in `env.py` encode this; do not widen them.

**The eyes' readout is one gather, not a fisheye pass and a pooling pass.**
`flyplay/retina_fast.py`, reached through `FlySim.ommatidia_readouts`, which
every caller in flyplay uses in place of `Simulation.get_ommatidia_readouts`.
FlyGym recomputes the same 512x450 coordinate transform on every call and
writes a whole image that the pooling immediately reads back; which source
pixel lands in which destination pixel is fixed by the retina, so the readout
is one accumulation. Both eyes with the render: 5.81 ms against 1.75 on a
pinned core. It is bit-identical, not merely close -- the fisheye copies
pixels rather than blending them, destination pixels whose source falls off
the image add `0 / hex_pxl_size` either way, and the additions happen in the
same ascending order (FlyGym's `prange` there is `parallel=False`) -- and
`EyeReadout` checks itself against FlyGym's own two passes when it is built,
raising rather than drifting after an upgrade. Checked again on 120 real
renders: every frame equal. What it is worth depends on how often the eyes are
read -- 10 s a room, fingerprints identical: `scented_sugar` 811 -> 775 ms of
wall time per simulated second at 10 Hz, `shadow_sugar` 1756 -> 1133 and
`drum_turning` 1998 -> 1176 at 100 Hz.

**Vision costs 25 ms per query** against 0.106 ms for a physics step. Hence
`vision_hz` decoupled from `action_hz` (20 Hz vs 100 Hz, last features held in
between) — that choice is what makes training take 30 min instead of 100.

**Index observations through `OBS_SLICES` / `obs_field()`**, never literals.
The layout is declared once in `OBS_LAYOUT`.

**A checkpoint's obs dim must match the env you build.** `--follow` reads
`out/rl/<run>/meta.json` to decide which env to construct; a stale or
hand-edited `meta.json` silently builds the wrong environment and the load
fails with a dimension mismatch.

**Spawn rotation must be a quaternion.** The freejoint rejects
`Rotation3D("euler", ...)`.

**`BlocksTerrainWorld` has no floor.** `_blocks_world()` adds a base plane;
without it the fly walks off the ~30 mm patch and falls into empty space,
which reads as a locomotion failure but is not.

### Conditioning (mushroom body)

**Steer on each odour's own valence, never the mixture's.** On a test trial the
two sources sit on opposite sides, so summed left-right asymmetries cancel and a
fly steering on (mixture valence × summed asymmetry) walks straight between them
whatever it has learned. `_turn_command()` weights each odour's gradient by that
odour's valence.

**Never place a source head-on.** Asymmetry straight ahead is 0.0006 against
~0.02 at 35°; no gain recovers it, and a fly that learned avoidance has no signal
to act on. Training sources use a random bearing inside the test cone.

**Keep odour intensity inside the Kenyon-cell working range.** The KC code holds
its shape (correlation 0.989) only over ~0.0016–0.11. Unclamped, a fly with its
head on a source sees 2.1 and the mushroom body files it as a different odour.
`ODOR_MIN_DISTANCE = 3.0` caps intensity at exactly 0.111.

**Divisive normalisation `r/(σ+Σr)` does nothing to the KC code.** The denominator
is a scalar and top-k is scale-invariant — sweeping σ changes zero cells. Do not
tune σ expecting invariance; it comes from the ORN stage.

**One trial loop.** `ConditioningExperiment.trial_steps()` is a generator that
`run_trial()` drains and the web viewer steps through. The viewer once carried a
copy, and the first steering change broke only the viewer. The colour task has
its own loop (`_colour_trial_steps`) behind the same entry point, kept separate
so the odour loop stays byte-identical -- checked by a fingerprint of paths,
valences and weight hashes before and after vision went in.

**Reversal is fixed by extinction, not by tuning `recovery`.** With passive
forgetting alone, reversal reaches PI −0.30 against +0.49 for acquisition. The
omission-of-punishment signal (`ConditioningConfig.extinction="avoid"`, an
MBON→DAN `feedback` in `MushroomBody.step`) brings it to −0.50. It must stay off
on punished trials — the model has no timing, so "no shock yet" and "no shock"
are indistinguishable within a trial.

**Wired to `approach`, the omission signal is an unstable positive feedback.**
Measured: an odour sharing 8% of Kenyon cells with the punished one starts at
valence −0.004 and doubles on every unpunished exposure to −0.65. Any change to
where `feedback` dopamine goes needs `13_mb_check.py` section 6 to pass first.

**Memories carry their config.** `save_memory` stores weights, wiring seed and
the full `ConditioningConfig`; `experiment_from_memory` rebuilds with the file's
wiring (never the caller's seed) and config. Weights on another wiring sit on
the wrong Kenyon cells silently, so `load_memory` refuses a wiring mismatch.
Filed memories live in `memories/`, outside `out/`, because sweeps clear their
output directories. Format 2 adds the visual cells: a smell-only memory loads
into a seeing fly with its visual synapses at rest, and a memory with visual
synapses is refused by a fly without vision rather than silently truncated.

### Vision in the mushroom body (colour task)

**Visual Kenyon cells are scaled, not summed to 1.** Each sense's code sums to 1
on its own; appended raw, 5 active visual cells would weigh what 100 olfactory
ones do and learn twenty times faster per synapse. `MushroomBody` multiplies the
visual code by `k_visual / k_olfactory` (0.05), so an active cell carries the
same activity whichever sense drove it (measured ratio 1.00). Consequence:
learned visual valence spans only ±0.05; steering and the viewer divide by
`mb.visual_scale` to get -1..+1.

**VPNs read only the row band 0.6–0.8 of each eye.** Measured over a black
floor: the rows nearest the horizon blend in sky (0.121/0.067) and the lowest
rows see the fly's own legs (up to 0.076). Pooled in, those offsets desaturated
dim colours and would let the colour pathway "see" brightness, breaking the
Vogt 2016 dissociation. In the band, blue and green read as exact mirror images
and a black floor reads 0.000.

**Hue and brightness compete for the same few winners.** With random wiring,
bright and 10x-dimmer green get nearly the same code (cosine 0.88) and a
brightness task is unlearnable; half the cells get one brightness input for that
reason (0.71). The trade-off table is in `visual_pathway.py`; do not rewire
without re-measuring it.

**Colour tiles are visual only and follow the fly.** `ColourFloor` moves the grid
by whole checker periods (two tiles) so the pattern is unchanged everywhere;
which colour the fly stands on comes from `colour_at` (the pattern), never from
geom positions. Repainting uses `mj_model.geom_rgba`, which reaches the eyes on
the next render without a rebuild.

**The visual front end answers from its last call while the eyes have not
changed.** In a 10 Hz room `Sandbox.step` hands the same readouts to
`MushroomBody.step` for nine more physics steps, so the pooling and the top-k
ran a hundred times a second on one picture: 13.5 ms of each simulated second
down to 1.9, fingerprint unchanged. `VisualFrontEnd.__call__` compares the
readouts by value, not by identity, so writing into a readouts array in place
still gives a fresh answer; a room where something moves reads new eyes every
step and never hits.

**Colour steering is the bilateral comparison alone.** Ceiling test with a
hand-set memory: bilateral gain 3 gave LI -0.56; adding kinesis dropped it to
-0.32 because walking faster eats turning authority. Numbers in
`colour_config`'s docstring.

**A colour trial renders the eyes itself** (inside `_colour_trial_steps`, at
`vision_hz`). The viewer draws `state.visual` rather than rendering again, which
would cost another 25 ms. Pooled, colour flies run at 3.7 wall s per simulated s
each against about 1.7 for odour flies.

**Four colour ablations are identities.** Shared dopamine, gamma-d block and
either DAN block leave the eyes' valence exactly zero, so the fly steers as a
frozen one does and must walk its exact path. They run on two seeds and the
summary checks paths; a mismatch is a bug, not noise.

### The sandbox room (`flyplay/room.py`, `flyplay/sandbox.py`, `09_web_viewer.py --sandbox`)

**Walking into a wall flips the fly.** Measured: 4 and 8 mm walls met head-on or
at 45 degrees flip it within 0.35–2.3 s; wall friction 0.1 or 0.01 does not help,
only a 75-degree graze slides along. Hence `ProximityReflex`, which turns before
contact (12 approaches, 0 flips, 0 contacts) and outranks every other behaviour.
A symmetric descending signal of 0 stops the fly upright (0.5 mm/s), which the
0.4 floor above does not forbid -- it is about asymmetric turns. `Walker.reset`
sets adhesion unconditionally, so a fly built with `adhesion=False` raises
KeyError.

**Everything placeable is pooled.** Collision pairs are fixed at `add_fly()`, so
`add_room` builds every obstacle, drop, zone, patch and odour marker up front,
parked at (900, 900), and placing one moves its `geom_pos`. `POOL_SIZES` is
therefore a hard cap on what the page can place.

**The room floor is dark on purpose.** On the grey arena floor, blue acquired
+0.66 valence in a fly that never set foot on it: the floor's own visual code
overlapped the patches' and was learned as context. Dark floor: overlap 0 on 16
seeds, blue 0.0.

**Shock dopamine is a 1.5 s pulse from each onset,** not the contact time. A fly
escapes in ~0.2 s; seven such shocks left blue at -0.10, too faint to steer by.

**A drop is tasted over the radius it is drawn at** (never under 1.5 mm; kept at
its bout-start value while being fed on). With a fixed 2.5 mm, 20, 100 and
500 nl drops gave step-for-step identical runs.

**Only the solids near the fly keep their bounding spheres.** The room's
walls are 104 mm long, so their spheres cover the whole floor and every leg is
tested against every wall on every step -- collision detection was 313.5 ms of
each simulated second, about two thirds of it walls. `Room.gate_bounds`, called
from `Sandbox.step` with the thorax position just before the physics, leaves a
real `geom_rbound` only on the walls and obstacles whose footprint is within
`GATE_MM` (8 mm) and shrinks the rest to **1e-6 -- not 0, which MuJoCo reads as
"this geom is a plane" and stops filtering**. 8 mm is the margin: the fly
reaches 3.0 mm from thorax to tarsus tip and moves under 0.3 mm in the 10 ms
between calls. **Anything that steps physics outside `Sandbox.step` must call
`Room.ungate` first.** `_settle_body` does, which covers `reset_fly`,
`return_home`, `move_fly`, `rest` and `restore`; without it a fly set down
beside a wall whose sphere is still 1e-6 walks through it without one contact.
Measured over 10 s a room, two seeds, gate and visual cache off against on,
every `qpos` and readout fingerprint identical: `t_maze` 1283/1235 -> 949/931
ms of wall time per simulated second, `scented_sugar` 979/991 -> 784/758,
`obstacle_field` 1022/1031 -> 811/823, `corridor` 1086/1079 -> 953/924, and -6
to -14% in `dead_end_maze`, `heat_maze`, `drum_turning` and `shadow_sugar`.
With the proximity reflex off and the fly driven into a wall, a block and
another wall -- 798 of 1400 steps in contact, 338 of them upside down --
`qpos` was identical at every step, with 19.2 of the 20 solids gated off on
average.

**Obstacles are resizable boxes, so their bounds must follow.** A wall is an
obstacle with long `hx`/`hy`; `Room._apply_params` rewrites `geom_size` *and*
`geom_rbound`/`geom_aabb`. Measured with a 40 mm wall hit 15 mm off centre: with
the bounds updated the fly stops at the face; with the compiled 4 mm bounds left
in place there is no contact and it walks straight through.

**Resetting the body rewinds `sim.time`, and the viewer paces on it.** Settling
a fly steps physics from the reset keyframe. "Back to the centre"
(`Sandbox.return_home`) keeps memory, hunger and tallies and restores the clock
afterwards; a new fly or a trial start really does restart time, so the viewer
resets its pacer (`_rewound`) -- otherwise the pacer's anchor sits in the future
and it never waits again.

**Shock counts are per visit, not per contact onset.** Contact is read from
planted feet and flickers as legs step; a new shock counts only after 1 s off
every zone (`shock_rearm_seconds`).

**Odour goes round obstacles, by path length** (`WalledOdorField`). Intensity is
still NeuroMechFly's inverse square, of the shortest path round the solids
(corners of the obstacle rectangles as graph nodes) rather than the straight
line. Where a source is in plain view the reading is bit-identical to the
straight field, so every open-room calibration stands (sandbox fingerprint
unchanged in scented_sugar and blue_shock). Measured, six flies per room, first
feed within 90 s: dead_end_maze 4/6 found (median 40 s) with odour through walls
against 6/6 (16.6 s) round them; wall_between, two_rooms, room_in_room and t_maze
about the same either way. A read costs 65-232 us (16 obstacles, 64 corners).
It is geometry, not diffusion: PLAN.md B-1's grid solution would change the
field in the open room too. A sealed box reads zero outside.

**The sandbox fly survives a restart; its synapses only onto the same wiring.**
`flyplay.session` writes the fly (every synapse, hunger, both clocks, room with
what is left of each drop, tallies, event log, central complex, random state)
every 30 s and on a clean exit, renames the state file into place, and appends
the trail instead of rewriting it (a day of trail is 6.9 MB). The session is
locked by an OS file lock, released however the viewer dies. State carries
`MushroomBody.wiring_signature()` -- a hash of every fixed matrix between the
senses and the synapses and the compartments' cell masks -- and a mismatch sets
the old state aside and starts a new fly: weights on other wiring would sit on
the wrong cells without an error. Killed with Stop-Process and restarted, a
viewer came back with the same room, drop contents, hunger, clock, values,
trail and results row.

### The sandbox's brain (`flyplay/olfactory.py` `ConnectomeFrontEnd`, `flyplay/sandbox.py`)

**Every addition is a `SandboxConfig` switch, and all of them off is the old
fly exactly.** `brain="random"`, `individuality=False`, `wall_following=False`,
`search_radius_start=12`, `central_complex=False`, `labellum_contact=False`
reproduces the sandbox fingerprint (scratch `fingerprint.py`, three presets, 600
steps) of the model before these changes, bit for bit; checked after each one.
Keep that true: it is how a change is shown to touch only what it claims to.

**Receptors are DoOR's, claws the hemibrain's.** A glomerulus responds to an
odorant as DoOR's consensus for its receptor (SFR subtracted, inhibition
clipped, unmeasured = 0); the room's vinegar is a blend (acetic acid 1, ethyl
acetate 0.5, acetoin 0.5 -- DM1 and VA2 need driving, Semmelhack & Wang 2009;
the proportions are a model choice). 3-octanol's Or13a response is left out as a
contaminant (Lüdke et al. 2025). The 1733 olfactory Kenyon cells keep their
hemibrain types and lobes; `wiring="sampled"` (default) redraws each cell's
glomeruli from its type's preferences with the cell's own claw count and synapse
counts, per seed, so flies differ as real ones do and twins share wiring.

**APL inhibits each lobe on its own.** One global APL gave gamma cells (7 claws,
averaging more glomeruli) 13-23% of the code for 35% of the cells, and the
gamma1pedc compartment a third of its learning. Per lobe (Amin et al. 2020:
APL acts locally), feedback strength 52 puts the median odour at 7.2% of each
lobe's cells (range 2.4-11.8%), each lobe carries exactly its share of the code,
and blocking APL (0.2) gives 3.1x the active cells with similar mixtures more
alike -- Lin et al. (2014)'s result. `13_mb_check.py` section 9 checks all of it.

**Compartments read their own lobes, rescaled.** gamma1pedc (approach_fast)
reads gamma cells, alpha3 (approach_slow) alpha/beta, beta'2+gamma4 (avoid_sweet)
alpha'/beta' and gamma, gamma5+alpha1 (avoid_nutrient) gamma and alpha/beta;
input is divided by the lobes' share (`Compartment.coverage`) so a novel odour
still reads `w0`, and `eta` is scaled by the cells a compartment reads over the
100 of the old code, so one pairing depresses about as far as before: one shock
pulse 0.30 against 0.22, five seconds of sucrose 0.56/0.29 against 0.53/0.27.

**Octanol and MCH now overlap, because their receptors do.** Both drive Or69a
(glomerulus D) hardest; receptor profiles cosine 0.47, Kenyon-cell codes 0.33
(0.16-0.46 over 8 flies) against 0.05 on random wiring. Learning generalises:
sucrose with octanol raised MCH to 45% of octanol's value (10% before). In the
room, condition A of sugar_odour gave test PI +0.90 (4 flies; +0.97 random) with
MCH at +0.50 (+0.04), and shock_odour +0.02 (+0.05). Vinegar stays apart from
both (cosine under 0.05). The conditioning tasks keep `OlfactoryFrontEnd`.

**Visual Kenyon cells are the hemibrain's 99 gamma-d and 60 alpha/beta-p** (159,
was 100), lobes attached. The dark room floor still shares no cell with blue or
green on 16 seeds.

**Motion vision runs only while something moves by itself** (`flyplay/
motion_vision.py`: Hassenstein-Reichardt correlators on ON and OFF signals,
Borst 2018's 250/50 ms filters). A turning drum or a shadow item switches the
eyes to 100 Hz; a still drum (a panorama) and every other room keep 10 Hz and no
motion output, as if an efference copy cancelled the fly's own flow -- at 100 Hz
every experiment would slow. The drum and the shadow are mocap bodies parked
under the floor plane: placing nothing, the fingerprint is unchanged (checked,
eye-dependent blue_shock included). `ROTATION_SIGNS` and the thresholds come
from rendered eyes with the fly held still (numbers in the module): a
counterclockwise drum gives negative image-column motion in both eyes, walking
forward opposite signs that cancel. The shadow detector reads net dimming, OFF
minus ON, over the upper field: with OFF alone the drum's stripes, which fill
that field beside a wall, read as shadows and froze flies (2-9 mm/s beside a
turning drum against 13-16 beside a still one).

**The central complex loses its bearings whenever the fly is put down.**
`CentralComplex.put_down` (every `reset_fly`, `return_home`, `move_fly`, `rest`
and restore) gives the compass a random heading and zeroes the integrator; only
a learned landmark bearing re-anchors it to a panorama. That is what makes a
place remembered in one trial findable in the next with landmarks and not
without, as in Ofstad et al. (2011) -- do not "fix" it by starting the compass at
the true heading. Ring input is where the panorama's dark features are
(`landmark_view`: ommatidia reading under 0.3 in the upper half of the field,
their mean bearing and how clear it is), with azimuths from
`flyplay/data/ommatidia_azimuth.npz` (rendered one drum stripe at a time: left
eye -12 to +135 degrees, right -135 to +14, about 180 ommatidia each). Three
richer forms failed first (module docstring): Hebbian ring-to-wedge weights
swung the estimate 100-200 degrees within seconds, a whole-panorama template
re-anchored three put-downs in six because walls fill the view near them, and
counting the grey walls as dark made the nearest wall the landmark. The place
memory it gives is weak: over 8 trials of 45 s, 4 flies, the landmark flies
reached the cool spot in 14.3 s in the last two trials against 27.5 s without
(every pair), but were already faster in the first two (25.4 s against 34.1 s);
a stronger pull (`VISUAL_GAIN` 1.0) made them slower, 39.2 s. `place_memory`
has not been piloted. Local
search now returns to where the integrator says the meal was, errors included.
A goal is set on relief -- cool floor after a second on hot -- and steered toward
only while the floor is hot. `Sandbox.cx_view` puts the believed goal and meal
on the map (star and cross).

**Heat, light and bitterness are dopamine and reflexes, not scenery.** A heat
zone (hidden, like shock plates) punishes above 30 C (`heat_drive`, half at 34),
turns the fly back when the antennae are 3 C warmer than the floor under it, and
speeds it up on hot floor; rectangles (`hx`/`hy`) tile a floor round a cool
spot. A light zone writes punishment or reward dopamine directly, with no sense
and no hunger gate (CsChrimson). Bitter scales a drop's sweetness by 1 - bitter,
punishes while tasted, and stops a fly starting a meal when little sweetness is
left. Drinking needs the labellum (the haustellum geom's centre) over the drop;
a fly whose labellum misses folds, creeps toward the middle and extends again.

**Individual flies differ, from their own generator.** `Individual.draw` uses
`default_rng([seed, 7907])`, leaving the behaviour stream alone, so twins in an
A/B set are the same individual: left/right saccade bias Beta(4.2, 4.2) (24.0%
of flies beyond 70/30, Buchanan et al. 2015: 23.5%), walking drive and wall
affinity with a log s.d. of 0.1 and 0.3 (model choices). Wall following holds a
wall alongside at 4.5 mm -- where the reflex's 25-degree ray no longer reaches
it -- and halves saccades along it.

**What the new behaviours measured** (4 flies per arm, same seeds, 14 pinned
workers; scratch `behaviour_calibrate.py`): time in the room's outer third
0.89 with wall following against 0.72 without, over 120 s (real flies 0.90-0.95,
Soibam 2012; gain 0.15 without the saccade halving gave only 0.82). Shadow
bursts (five passes a second apart, every 10 s) raised walking speed to 16.1
mm/s from 14.1 with 2-8 s of freezing in 60 s, Gibson et al. (2015)'s direction;
before freezing needed 5 quiet seconds, freezes chained through each burst and
flies slowed to 9 mm/s. Bitter sugar (0.6) was fed on 9.0 s against 12.8 and
left vinegar at -0.22 against +0.75. Three minutes with octanol under a
punishing light left octanol at -0.13 to -0.23 and MCH at -0.02 to -0.13.
A drum turning counterclockwise at 60 deg/s turned all four flies with it at
+17 to +21 deg/s with `optomotor_gain` 1.0 (their still-drum twins -20 to +19,
circling the walls either way); at 0.3 two of the four kept circling clockwise.
The shadow numbers above were measured at 0.3. Drinking needs the labellum
(`labellum_contact`, on in all of these runs: the bitter test's plain sugar was
fed on for 8.9-16.8 s); on against off was not measured.

**The fly's eyes see its own legs, so colour goes on the viewer's copy only.**
Built with `colorize=True`, 350 of 1442 ommatidia changed, VPN outputs by up to
0.42, and the visual Kenyon-cell code differed on 57% of samples (same path,
nothing learned). The sandbox builds plain and `_render` paints the scene geoms
after `update_scene` (`display_colours`). The colour conditioning task still
builds coloured; its results predate this finding.

**The proboscis is a motor plus a joint spring, not a position servo.**
`Simulation.set_actuator_inputs` wants one input per actuator of a type, and
`apply_locomotion_action` sends 42, so a proboscis POSITION actuator breaks
walking. At the legs' stiffness of 0.05, gravity sagged it to (-0.60, +0.36) rad
with no torque; at 5 it sags 0.01. Only `build(proboscis=True)` adds it --
every other scene's model is unchanged. Choose its pose by eye as well as by
numbers: minimising the tip's height alone gave a 47 deg kink with the
labellum pointing back at the legs, which looked wrong; the straight pose
(0.7 deg) touches the drop just as well. Timing matters as much as pose --
out only once the fly has stopped (0.34 mm/s median), sipping, folded before
walking off.

**`import flyplay.build as B` gives the `build` function, not the module:**
`flyplay/__init__.py` re-exports it under the same name. Setting a module
constant through `B` silently does nothing; use
`importlib.import_module("flyplay.build")`.

### Web viewer page

**Canvases live absolutely positioned inside `.plot`.** `fitCanvases()` writes
the displayed size back into the canvas attributes; in normal flow that is a
ratchet where a card can grow but never shrink — measured, one card grew to
465 px over the video while another fell to 13 px.

**Decide the layout on every `/state`, not the first.** The HTTP server answers
before the simulation thread publishes, so the first `/state` can be empty.

**The page is a plain (non-raw) Python string.** A `\n` in its JavaScript
becomes a real newline inside a JS string literal and breaks the whole script.
Write JS without backslashes (`String.fromCharCode(10)`), and check an edit by
extracting the script and running `node --check` on it.

**`classList.toggle(name, force)` needs a real boolean.** With `undefined` it
flips instead: in select mode with nothing selected every palette options row
opened at once and the map shrank to 175 px.

**Nothing above the map may change height on selection.** Clicking the same spot
again cycles through overlapping items, which only works if the map stays put:
selecting once pushed it down 30 px, then 17 px (a wrapped preset row), and the
second click landed 30 mm away. The options rows sit in a fixed 54 px box and
the preset row never wraps.

**Undo records hold each edit's inverse, never a copy of the room.** A copy
would also refill every drop the fly has eaten since. Removed items come back
under their old id (`Room.place(item_id=...)`), so older records that name them
still find them; a record whose item the fly has since eaten is skipped.
Only room edits are recorded; a trial start re-creates every item and drops the
history, and scripts that rebuild a room through page commands send
`history_clear` so a Ctrl+Z cannot take the rebuild apart.

**One drag or slider pull is one undo step because the page tags it with a
gesture id.** The server's fallback merges same-kind edits closer than 1.5 s --
and a POST from Python's urllib to `localhost` took 2.0 s here (IPv6 tried
first), which split a five-step drag into five records. Test scripts use
`127.0.0.1`.

**`FlyServer` methods are shared by every mode.** Sandbox trial methods named
`_start_trial`/`_finish_trial` were silently replaced by the conditioning
methods of the same name defined later in the class (TypeError on the first
trial). Sandbox-only names carry `_sb_`.

**Header values hold still.** The sandbox header has no odour / nearest-sugar /
wall-contact line (the server does not compute those fields there): its values
changed length on every poll, re-wrapped the header and shook the layout under
it. The remaining values have fixed `min-width`s for the same reason.

**The video keeps 240 px, and the card under it shrinks.** The A/B pane first
took 393 of the column's 440 px at 1366x768 and left the video 39 px. The card
scrolls instead; its grid has a fixed height.

**A/B experiments live in a dialog over the whole page, one step per screen.**
The pane under the video held a set picker, a record picker, room previews and a
results grid in 192 px, with the prediction offered as "A가 더 작다", and the user
found it too hard to use. The lab (`labOpen`/`labGo`) walks through question ->
hypothesis and start -> result and replays; the tab under the video keeps only
an open button and the last three runs.

**A tab plays one video stream, and only while it is on screen.** Each stream
holds one of the browser's six connections to this server for as long as it
plays, across all tabs. `syncStreams()` decides every `<img>`'s `src` from what
is visible: `#view` unless the tab is hidden or the lab is open, the lab's own
video only on its result screen, the eye mosaic only in modes that show it. The
replay bar is a single element moved into the lab's video and back, so its
handlers stay bound.

**The server draws only for watchers.** `/stream` and `/eyes` connections are
counted (`FlyServer.watch`); with none, `_render` skips the scene render and
`_render_eyes` the mosaic, and `/neuro`'s payload is not built without a poll in
the last 2 s. Measured on the sandbox: a 1280x720 frame with its JPEG 17.3 ms,
the mosaic 2.2 ms, one 10 ms step 13.1 ms -- at 25 fps the two display costs
took half of each wall second from the simulation thread. Panels of other modes
start `display:none` in the markup: the eye mosaic, odour trace and attribution
panels showed for a moment in the sandbox before the first `/state` arrived. The
page's draw functions return early for canvases `onScreen` rejects.

**The server answers `exp_start` before the run exists** (commands queue for the
sim thread; `/sandbox` returns 204). The lab recognises its run as the first
experiment of that set missing from the list it had before the click, and gives
up with a message after 15 s -- which is what the 10 s duplicate guard looks like.

**Hypotheses are the set's own sentences.** `ExperimentSet.outcomes` words "A<B"
and "A>B" for each set; "same" is generic. A preference index names its sides
through `pi_sides` (+1 octanol / -1 MCH, +1 blue / -1 green): the metric's own
sentence points at the report's "냄새 선호 기준" column, which the lab does not show.

**The trail lives on the server, and `/state` carries only its newest 50
points.** `Sandbox.trail` keeps a day at 10 Hz as float32 pairs; the page
fetches older points once from `/trail` (base64) and appends the rest from each
poll. `Trail.epoch` changes when the trail starts over, which is how the page
knows to drop its copy. Sending the whole trail every 150 ms would be 36,000
points an hour.

### Background A/B experiments (`flyplay/experiment.py`, `flyplay/report.py`, `scripts/17_run_experiment.py`)

**Idle thread pools spin unless told to sleep.** One sandbox viewer process used
14.7 cores: numba runs FlyGym's retina as parallel loops ten times a second on
the OpenMP layer, whose workers busy-wait longer than the gap between loops,
and numpy's OpenBLAS keeps a thread per core. It made every other measurement on
the machine wrong (a CPU pool first read 3.75x total, 6.5x once fixed).
`09_web_viewer.py` and `17_run_experiment.py` set `OMP_WAIT_POLICY=PASSIVE`,
`KMP_BLOCKTIME=0` and `OPENBLAS_NUM_THREADS=1` **before** numpy and numba are
imported: same 13.7 ms step, 4.7 cores down to 1.1. A new script that builds a
`Sandbox` needs the same lines at its top. Before timing anything, check what
other Python processes are burning.

**Fly k of A and fly k of B are twins.** They share the sandbox seed (wiring and
exploration random numbers) and the start headings, which `FlyRun` draws from
`(protocol seed, fly)` rather than from the sandbox's generator. Without that a
difference between conditions would mix the manipulation with individual luck.
Running a set again from the viewer uses a new protocol seed: new flies.

**The experiment directory is the only interface.** Workers append one JSON
line per finished trial to their own `records/c<condition>_f<fly>.jsonl`, never
a shared file; `status.json` is written whole and renamed; a file named `STOP`
ends the run after the running trial, which is recorded as stopped. The viewer,
the report and the CLI read nothing else, so a GPU runner could replace the
workers without touching them.

**A replay poses the viewer's own model.** Every sandbox seed compiles the same
model (nq 75, identical geom and joint names), so recorded `qpos` from a worker
replays on the viewer 0.006 mm from the recorded thorax. `_replay_start` saves
the live fly with `mj_getState(mjSTATE_INTEGRATION)` and its room as item specs;
nothing steps during a replay, and any command that touches the live fly or its
room ends the replay first (`REPLAY_OPS`). Replays are 50 Hz (legs cycle at
about 12 Hz) and cost 0.85 MB per simulated minute, so only flies up to
`replay_flies` are recorded.

**Preference indices count sides, not visits.** `colour_pi` and `odour_pi` credit
each 10 Hz sample to whichever stimulus is nearer, as a T-maze scores the arm the
fly is in; a trial that never came within reach of either still counts. Being
*near* a source (`near_*_pct`, 10 mm) is reported separately.

**A latency that never ended counts as the trial's length** when conditions are
compared (`metric_value`): a fly that never fed in 60 s is slower than one that
fed at 50 s, not missing. The report says so on every latency chart.

**Conditions may run different numbers of trials** (four trainings against one):
report series are padded to the longest, or the chart indexes past the shorter
(IndexError, found writing the `more_training` report).

**No verdict before three pairs.** With one or two pairs of flies every split is
a coin flip (sign test p = 1.0 or 0.5), and a one-pair pilot read "가설이 맞았습니다".
Beyond that the report calls a hypothesis supported when the mean difference
exceeds the measure's `tolerance` and 70% of pairs agree -- wording for a
middle-school report, printed beside the sign test's probability.

**The verdict says how big, and how many flies would make it believable.**
`paired_effect` is d_z (mean pair difference over its s.d., capped at 99 so the
payload stays JSON); `pairs_needed` is the fewest pairs for which the sign test
would come out under 5% in 8 of 10 reruns, taking this run's share of agreeing
pairs pulled toward a coin flip ((agree+1)/(n+2)) and its tie rate. 3 of 3
agreeing need 20 pairs, 8 of 8 need 12, 6 of 8 need 49.

**A set must ask something the model does not already state.** "영양 없는 단맛도
배를 채울까?" was removed: hunger falls only with calories by construction, so
the answer was written before the run (the user's call). Runs of a removed set
still list and report; they cannot be run again.

**The lab's time estimates use the newest runs' own speed**: the median over the
last three finished runs of simulated seconds over trial wall time. One constant
went stale twice in an evening, and one run slowed by other work read 0.26x
against 0.41-0.46x for its neighbours.

**A set is only as good as its design.** Each set's conditions must actually let
the fly meet the stimulus (sugar that is found, a shock zone that is walked
onto) or both arms read the same: a lone odourless drop was never found in 60 s
by either arm, and vinegar sugar was found in 3 s from the first trial, leaving
nothing to learn. Which sets were piloted, and which were redesigned afterwards
without a pilot, is in PLAN.md E. **The user starts experiments**: do not run
pilots or sweeps on your own -- offer the command.

**Experiment workers must be pinned to cores** (PERFORMANCE_PLAN.md X1). A
thread holding an OpenGL context on the RTX is kept on the 4 P cores by
Windows, however busy they are, so 14 workers shared 4 cores. Each worker given
its own core with `psutil.Process().cpu_affinity` (P 1-3, E 4-11, LP-E 12-14)
ran the same 120 s wall-following flies in 199-309 s of wall time against
409-426 s unpinned, with identical results. `17_run_experiment.py` pins them
(`pin_worker`, one core each counting up from 1 so core 0 is left to the viewer
and the desktop, no pinning at all when there are not enough cores to go
round), and gives the workers `NUMBA_NUM_THREADS=1` because a pinned worker's
16 OpenMP threads would share its one core. `--no-pin` turns it off, which is
how the two are compared.

**The GPU does not win at this scale.** Measured here: MuJoCo-Warp on the RTX
5060 walks 4096 flies at 12.8x real time in total *for the physics alone*,
against 6.5x for 14 CPU processes running the whole fly. Porting the controller,
the mushroom body and the eyes to batched form, and re-validating walking without
noslip iterations (MJWarp drops them), is what the missing 2x would cost.
The sandbox's own model (room pools, 756 leg-vs-wall pairs, proboscis) walks as
fast as FlyGym's plain fly: 64/256/1024 worlds at 5.0/9.2/12.5x total, 4.7/8.8/
12.2x with a 500 Hz CPU round trip. 2048 worlds filled the laptop GPU's 8 GB
(7.9 GB used) and had not finished 0.2 s in 16 min, so about 1024 is the ceiling
here; below about 100 flies the CPU pool is faster. MJWarp also refuses a pair
margin while MULTICCD is on (the model's pairs carry 0.001): set margins to 0
for a GPU copy. `GPUSimulation` has no contact forces, per-world reset or eye
readouts -- the CPU methods read `mj_data`, which GPU stepping never updates.

## Windows specifics

- **`pip install -r requirements.txt` fails in a deep folder.** Long paths are
  not enabled here: a fresh clone in this session's scratchpad put setuptools'
  test data at 275 characters, and pip stopped with `OSError: [Errno 2] No such
  file or directory`. The same install at `%TEMP%\fv\flyplay` (182 characters)
  took 199 s. Verify releases from a short path.
- **Rendering goes to the RTX through `SHIM_MCCOMPAT`, set in
  `flyplay/__init__.py`.** Left alone, MuJoCo's OpenGL (the fly's eyes, the
  viewer's video) ran on "Intel(R) Graphics". The NVIDIA Optimus driver reads
  `SHIM_MCCOMPAT=0x800000001` when a context is created, so it steers this
  process and its children only -- not every Python program, as the Windows
  per-app Graphics setting on `python.exe` would. Measured: an eye render and
  read-back 10.9-16.5 ms on Intel, 0.9-1.9 ms on the RTX; the viewer's 1280x720
  frame with JPEG 17.3 ms to 8.5 ms; with 14 experiment workers busy, one fly's
  eyes took 349-536 ms of a simulated second on Intel and 85-95 ms on the RTX.
  The GPUs rasterise colour edges differently: over 114 fixed views the readouts
  differed by at most 0.022, and 17 of 228 eye views got a visual Kenyon-cell
  code differing in 1 of 5 cells. `10_sense_check.py` passes on both, but the
  fingerprint changes; runs started before 2026-09-17 21:48 rendered on Intel.
  `SHIM_MCCOMPAT=0x800000000` renders on Intel again.
- **Git Bash `sed -i` rewrote a CRLF file with LF endings** (flyplay/room.py,
  every line a diff). Edit through a script that keeps the file's endings, or
  check `file <path>` afterwards.
- **Console output must be ASCII.** The console is cp949; an em dash in a
  `print` raises `UnicodeEncodeError` and kills the process. Korean text in
  `.md` files is fine — this applies to stdout only.
- **Don't use PowerShell here-strings for Python.** `@'...'@` mangles quoting.
  Write a `.py` file to the scratchpad and run it.
- **Training runs as two processes**: `.venv\Scripts\python.exe` (launcher) →
  `Python312\python.exe` (actual training) → 6 `SubprocVecEnv` workers. Killing
  the launcher alone stops nothing and orphans the workers.
- **`taskkill /T` is permission-denied here**; walk the tree via
  `Win32_Process` `ParentProcessId` and `Stop-Process -Force` leaf-first.
- **`Get-Process python` has returned empty while training was running.** Use
  `Get-CimInstance Win32_Process -Filter "Name LIKE '%python%'"` instead.
- **Editing a running script changes nothing** — Python holds the code it
  loaded at start.

## Conventions

**Prose is Korean, code is English.** `README.md`, `RESEARCH.md`, `STATUS.md`
and replies to the user are in Korean; all identifiers, docstrings and comments
are in English. Keep both.

**Comments carry measurements, not intentions.** The house style explains *why*
a constant has its value by citing what was actually measured — "0.9 mm drops a
tarsus into a gap and the fly flips; 1.2 mm clears it", "25 ms per call, which
is 236 physics steps' worth". When you change a tuned constant, measure and say
so. When you cannot measure, say that instead of guessing.

**Verify a task is learnable before training it.** Both RL tasks were first
solved with a hand-written control law (proportional steering; the paper's
`ΔI = (I_L − I_R) / mean` chemotaxis rule). Those baselines are recorded in the
script docstrings and are what a trained policy must beat.

## Where things land

```
out/rl/<run>/  model.zip  meta.json  checkpoints/ppo_<n>_steps.zip
out/rl/tb/     tensorboard logs
out/experiments/<stamp>-<set>/  protocol.json  status.json  records/c<c>_f<fly>.jsonl
                                replay/c<c>_f<fly>_t<seq>.npz  trials.csv  report.html
```

Checkpoints are not disposable: `09_web_viewer.py` replays all of them to build
its sensory-attribution history, so deleting them erases that record. The same
goes for experiment directories: the viewer lists, replays and reports from them.

`RESEARCH.md` is a survey of the wider ecosystem and a ranked list of research
directions (easy / medium / hard) — consult it before proposing new work, and
add to it rather than duplicating.
