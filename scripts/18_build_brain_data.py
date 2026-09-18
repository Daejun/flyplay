"""Build the small data tables the olfactory front end reads, from published data.

    python scripts/18_build_brain_data.py --download   # fetch the sources (46 MB), then build
    python scripts/18_build_brain_data.py              # build from data/raw/

Two sources, both downloaded into ``data/raw/`` (git ignores it) and reduced to
tables small enough to commit under ``flyplay/data/``:

**DoOR 2.0.1** (Münch & Galizia 2016, Sci Rep 6:21841; CC BY-SA 4.0): how each
olfactory receptor type responds to each odorant, merged from many studies into
one 0-1 scale per responding unit. The consensus values have the spontaneous
rate (the ``SFR`` row) subtracted, so negative means inhibition. Written:
``flyplay/data/door_odorants.csv``, one row per odorant the model uses, one
column per antennal-lobe glomerulus, blank where DoOR has no measurement.

**hemibrain v1.2** (Scheffer et al. 2020, eLife 9:e57443; CC BY 4.0), with the
uniglomerular projection neurons' glomeruli from Schlegel et al. 2021 (eLife
10:e66018; ``flyconnectome/hemibrain_olf_data``, MIT): every Kenyon cell of one
right mushroom body, its type, and how many synapses it receives from the
projection neurons of each glomerulus. Written: ``flyplay/data/hemibrain_kc.npz``.

Measured while building (printed again on every run): 51 glomeruli with
uniglomerular PNs, 48 of them with a DoOR unit (none for DL2v's own receptor
subset, VA7m, VM6); 1927 Kenyon cells, 1761 of them with PN input.

One correction to DoOR: 3-octanol's response in Or13a (glomerulus DC2, 0.58)
is left out. Commercial 3-octanol carries 1-octen-3-ol, Or13a's best ligand,
and the response came from that (Lüdke et al. 2025, eLife 13:RP99513).

Receptor -> glomerulus is DoOR's own mapping (`door_mappings.csv`): its ``code``
column, else its ``glomerulus`` column when that names one glomerulus. Units
DoOR marks larval only are left out. ac3A maps to "DL2d/v" and feeds both.
Several units on one glomerulus are averaged over those measured.
"""

from __future__ import annotations

import argparse
import tarfile
import urllib.request
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

ROOT = _bootstrap.ROOT
RAW = ROOT / "data" / "raw"
OUT = ROOT / "flyplay" / "data"

SOURCES = {
    RAW / "door" / "door_response_matrix.csv":
        "https://raw.githubusercontent.com/ropensci/DoOR.data/v2.0.1/data/door_response_matrix.csv",
    RAW / "door" / "door_mappings.csv":
        "https://raw.githubusercontent.com/ropensci/DoOR.data/v2.0.1/data/door_mappings.csv",
    RAW / "door" / "odor.csv":
        "https://raw.githubusercontent.com/ropensci/DoOR.data/v2.0.1/data/odor.csv",
    RAW / "hemibrain" / "FIB_uPNs.csv":
        "https://raw.githubusercontent.com/flyconnectome/hemibrain_olf_data/master/FIB_uPNs.csv",
    RAW / "hemibrain" / "exported-traced-adjacencies-v1.2.tar.gz":
        "https://storage.googleapis.com/hemibrain/v1.2/exported-traced-adjacencies-v1.2.tar.gz",
}
ADJACENCY = RAW / "hemibrain" / "exported-traced-adjacencies-v1.2"

#: Responses DoOR has that are not the odorant's own: (odorant key, unit).
CONTAMINATED = {("octanol", "Or13a")}
#: The odorants the model uses, by DoOR name: the components of the room's
#: odours (`flyplay.olfactory.ODORANT_BLENDS`) and a few for stimuli to come.
ODORANTS = {
    "acetic_acid": "acetic acid",
    "ethyl_acetate": "ethyl acetate",
    "acetoin": "3-hydroxy-2-butanone",
    "butanedione": "2,3-butanedione",
    "octanol": "3-octanol",
    "mch": "4-methylcyclohexanol",
    "benzaldehyde": "benzaldehyde",
    "isoamyl_acetate": "isopentyl acetate",
    "ethyl_butyrate": "ethyl butyrate",
    "geranyl_acetate": "geranyl acetate",
    "octenol": "1-octen-3-ol",
    "co2": "carbon dioxide",
}


def download() -> None:
    for path, url in SOURCES.items():
        if path.exists():
            print(f"have {path.relative_to(ROOT)}")
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"fetching {url}")
        urllib.request.urlretrieve(url, path)
    if not (ADJACENCY / "traced-total-connections.csv").exists():
        with tarfile.open(RAW / "hemibrain" / "exported-traced-adjacencies-v1.2.tar.gz") as tar:
            for name in ("traced-neurons.csv", "traced-total-connections.csv", "README"):
                tar.extract(f"exported-traced-adjacencies-v1.2/{name}", RAW / "hemibrain", filter="data")


def glomeruli_of_units(mappings: pd.DataFrame, units, known: set[str]) -> dict[str, list[str]]:
    """DoOR responding unit -> the hemibrain glomeruli it drives."""
    rows = mappings.drop_duplicates("receptor").set_index("receptor")
    out: dict[str, list[str]] = {}
    for unit in units:
        if unit not in rows.index or rows.loc[unit, "adult"] is False or str(rows.loc[unit, "adult"]) == "False":
            continue
        code, glomerulus = rows.loc[unit, "code"], rows.loc[unit, "glomerulus"]
        if isinstance(code, str) and code in known:
            names = [code]
        elif isinstance(glomerulus, str) and glomerulus in known:
            names = [glomerulus]
        else:
            names = []
        if glomerulus == "DL2d/v":
            names = ["DL2d", "DL2v"]
        if names:
            out[unit] = names
    return out


def build_door(glomeruli: list[str]) -> pd.DataFrame:
    matrix = pd.read_csv(RAW / "door" / "door_response_matrix.csv", sep=";", index_col=0)
    mappings = pd.read_csv(RAW / "door" / "door_mappings.csv", sep=";", index_col=0)
    odor = pd.read_csv(RAW / "door" / "odor.csv", sep=";", index_col=0)
    responses = matrix.drop(index="SFR") - matrix.loc["SFR"]
    units = glomeruli_of_units(mappings, matrix.columns, set(glomeruli))
    by_name = odor.set_index("Name")["InChIKey"]
    rows = []
    for key, name in ODORANTS.items():
        inchikey = by_name[name]
        row = responses.loc[inchikey]
        values = {}
        for g in glomeruli:
            measured = [row[u] for u, gs in units.items()
                        if g in gs and pd.notna(row[u]) and (key, u) not in CONTAMINATED]
            values[g] = round(float(np.mean(measured)), 3) if measured else np.nan
        rows.append({"key": key, "name": name, "inchikey": inchikey, **values})
    covered = sorted({g for gs in units.values() for g in gs})
    print(f"DoOR: {len(units)} adult units on {len(covered)} of {len(glomeruli)} glomeruli; "
          f"none on {sorted(set(glomeruli) - set(covered))}")
    return pd.DataFrame(rows)


def build_kc() -> dict[str, np.ndarray]:
    neurons = pd.read_csv(ADJACENCY / "traced-neurons.csv")
    upn = pd.read_csv(RAW / "hemibrain" / "FIB_uPNs.csv")
    connections = pd.read_csv(ADJACENCY / "traced-total-connections.csv")
    kcs = neurons[neurons["type"].astype(str).str.startswith("KC")].sort_values(["type", "bodyId"])
    glomeruli = sorted(str(g) for g in upn["glomerulus"].unique())
    edges = connections[connections.bodyId_pre.isin(upn.bodyid) & connections.bodyId_post.isin(kcs.bodyId)]
    edges = edges.merge(upn[["bodyid", "glomerulus"]], left_on="bodyId_pre", right_on="bodyid")
    table = edges.pivot_table(index="bodyId_post", columns="glomerulus", values="weight",
                              aggfunc="sum", fill_value=0)
    table = table.reindex(index=kcs.bodyId, columns=glomeruli, fill_value=0)
    synapses = table.to_numpy().astype(np.uint16)
    print(f"hemibrain: {len(glomeruli)} glomeruli, {len(upn)} uniglomerular PNs, {len(kcs)} Kenyon cells, "
          f"{int((synapses.sum(axis=1) > 0).sum())} with PN input, {int(synapses.sum())} PN->KC synapses")
    return {"glomeruli": np.array(glomeruli), "kc_body_id": kcs.bodyId.to_numpy().astype(np.int64),
            "kc_type": kcs["type"].astype(str).to_numpy().astype(str), "pn_synapses": synapses}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--download", action="store_true", help="Fetch missing sources into data/raw first.")
    args = parser.parse_args()
    if args.download:
        download()
    OUT.mkdir(parents=True, exist_ok=True)
    kc = build_kc()
    np.savez_compressed(OUT / "hemibrain_kc.npz", **kc)
    door = build_door(list(kc["glomeruli"]))
    door.to_csv(OUT / "door_odorants.csv", index=False, lineterminator="\n")
    print(f"wrote {OUT.relative_to(ROOT) / 'hemibrain_kc.npz'} "
          f"({(OUT / 'hemibrain_kc.npz').stat().st_size / 1024:.0f} KB) and "
          f"{OUT.relative_to(ROOT) / 'door_odorants.csv'} ({(OUT / 'door_odorants.csv').stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
