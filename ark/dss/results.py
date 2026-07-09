"""Turn a solved OpenDSSDirect.py circuit's results into tidy pandas DataFrames.

Every function here takes the `opendssdirect` module itself, not a wrapper
object, and reads the circuit's currently-solved state through
OpenDSSDirect's own `First()`/`Next()` iteration pattern. :class:`ark.dss.circuit.Circuit`
exposes these as methods (`circuit.bus_voltages()`, etc.) for everyday use;
call these module-level functions directly only when working with the raw
`opendssdirect` module. Long/tidy output (one row per bus-phase or
line-phase, not one column per phase) throughout, so results plot directly
through `ark.plot` by faceting or coloring on the `phase` column, the same
convention every other chapter's DataFrames already use.
"""

from __future__ import annotations

from types import ModuleType

import pandas as pd

# OpenDSS's own singular class-name prefix for `Circuit.SetActiveElement`,
# keyed by the plural collection name callers pass to `element_powers`.
_ELEMENT_PREFIXES = {
    "loads": "Load",
    "lines": "Line",
    "transformers": "Transformer",
    "pvsystems": "PVSystem",
}


def bus_voltages(dss: ModuleType) -> pd.DataFrame:
    """Per-phase bus voltage magnitude and angle for every bus in the circuit.

    Args:
        dss: An `opendssdirect` module pointed at a solved circuit (see
            :func:`ark.dss.circuit.load_circuit`).

    Returns:
        One row per (bus, phase): `bus`, `phase`, `vmag_pu`, `vangle_deg`,
        `kv_base` (line-to-neutral kV).

    Examples:
        >>> dss = load_circuit("simple_lv/Master.dss")
        >>> bus_voltages(dss).query("bus == 'b'")["vmag_pu"].round(3).tolist()
        [0.979, 0.979, 0.979]
    """
    rows = []
    for name in dss.Circuit.AllBusNames():
        dss.Circuit.SetActiveBus(name)
        kv_base = dss.Bus.kVBase()
        nodes = [n for n in dss.Bus.Nodes() if n != 0]
        mag_ang = dss.Bus.puVmagAngle()
        for phase, node in enumerate(nodes, start=1):
            vmag_pu, vangle_deg = mag_ang[2 * (node - 1)], mag_ang[2 * (node - 1) + 1]
            rows.append(
                {
                    "bus": name,
                    "phase": phase,
                    "vmag_pu": vmag_pu,
                    "vangle_deg": vangle_deg,
                    "kv_base": kv_base,
                }
            )
    return pd.DataFrame(rows)


def line_currents(dss: ModuleType) -> pd.DataFrame:
    """Per-phase current magnitude for every line in the circuit.

    Args:
        dss: An `opendssdirect` module pointed at a solved circuit.

    Returns:
        One row per (line, phase): `line`, `bus1`, `bus2`, `phase`,
        `current_a`.
    """
    rows = []
    if not dss.Lines.First():
        return pd.DataFrame(rows)
    while True:
        name = dss.Lines.Name()
        bus1, bus2 = dss.Lines.Bus1(), dss.Lines.Bus2()
        n_phases = dss.Lines.Phases()
        currents = dss.CktElement.CurrentsMagAng()
        for phase in range(1, n_phases + 1):
            rows.append(
                {
                    "line": name,
                    "bus1": bus1,
                    "bus2": bus2,
                    "phase": phase,
                    "current_a": currents[2 * (phase - 1)],
                }
            )
        if not dss.Lines.Next():
            break
    return pd.DataFrame(rows)


def line_losses(dss: ModuleType) -> pd.DataFrame:
    """Active and reactive power loss for every line in the circuit.

    Args:
        dss: An `opendssdirect` module pointed at a solved circuit.

    Returns:
        One row per line: `line`, `bus1`, `bus2`, `p_loss_kw`, `q_loss_kvar`.
    """
    rows = []
    if not dss.Lines.First():
        return pd.DataFrame(rows)
    while True:
        name = dss.Lines.Name()
        p_loss, q_loss = dss.CktElement.Losses()
        rows.append(
            {
                "line": name,
                "bus1": dss.Lines.Bus1(),
                "bus2": dss.Lines.Bus2(),
                "p_loss_kw": p_loss / 1000.0,
                "q_loss_kvar": q_loss / 1000.0,
            }
        )
        if not dss.Lines.Next():
            break
    return pd.DataFrame(rows)


def element_powers(dss: ModuleType, element_type: str) -> pd.DataFrame:
    """Per-phase active and reactive power for every element of one type.

    Args:
        dss: An `opendssdirect` module pointed at a solved circuit.
        element_type: One of `"loads"`, `"lines"`, `"transformers"`,
            `"pvsystems"`.

    Returns:
        One row per (element, phase): `name`, `phase`, `p_kw`, `q_kvar`.

    Raises:
        ValueError: If `element_type` isn't one of the supported collections.
    """
    if element_type not in _ELEMENT_PREFIXES:
        raise ValueError(f"element_type must be one of {sorted(_ELEMENT_PREFIXES)}, got {element_type!r}")
    prefix = _ELEMENT_PREFIXES[element_type]
    collection = getattr(dss, element_type.capitalize())

    rows = []
    if not collection.First():
        return pd.DataFrame(rows)
    while True:
        name = collection.Name()
        dss.Circuit.SetActiveElement(f"{prefix}.{name}")
        powers = dss.CktElement.Powers()
        # CktElement.Powers() is one (P, Q) pair per *conductor*, not per
        # phase: a single-phase load has 2 conductors (phase + neutral
        # return), and the neutral entry is always zero power. NumPhases()
        # is the true electrical phase count; only the first that many
        # (P, Q) pairs correspond to real phases.
        n_phases = dss.CktElement.NumPhases()
        for phase in range(1, n_phases + 1):
            rows.append(
                {
                    "name": name,
                    "phase": phase,
                    "p_kw": powers[2 * (phase - 1)],
                    "q_kvar": powers[2 * (phase - 1) + 1],
                }
            )
        if not collection.Next():
            break
    return pd.DataFrame(rows)


def circuit_summary(dss: ModuleType) -> dict:
    """A one-shot scalar health check for a solved circuit.

    Args:
        dss: An `opendssdirect` module pointed at a solved circuit.

    Returns:
        Dict with `converged`, `n_buses`, `n_lines`, `n_loads`,
        `n_transformers`, `total_p_loss_kw`, `total_q_loss_kvar`.
    """
    p_loss, q_loss = dss.Circuit.Losses()
    return {
        "converged": dss.Solution.Converged(),
        "n_buses": dss.Circuit.NumBuses(),
        "n_lines": dss.Lines.Count(),
        "n_loads": dss.Loads.Count(),
        "n_transformers": dss.Transformers.Count(),
        "total_p_loss_kw": p_loss / 1000.0,
        "total_q_loss_kvar": q_loss / 1000.0,
    }
