"""Real DER mitigation levers, as reusable, per-circuit functions.

Chapter 2's own `run_penetration()` built Volt-Watt and Volt-VAr inline,
one-off, inside a single feeder-wide penetration sweep. Promoted here so a
different customer, a different scenario, or a different call site can
apply the same real control without copying that code again. Storage is a
genuinely new addition: Chapter 2 only ever demonstrated a storage
element's charge/discharge state, never checked whether storage actually
fixes a voltage violation the way Volt-Watt and Volt-VAr were checked.

Every function here takes an already-built :class:`ark.dss.circuit.Circuit`
and issues OpenDSS commands against it; none of them solve the circuit
themselves, matching `Circuit`'s own pattern of leaving `solve_daily()` to
the caller.
"""

from __future__ import annotations

import numpy as np

from ark.dss.circuit import Circuit


def apply_volt_watt(circuit: Circuit) -> None:
    """Feeder-wide Volt-Watt inverter control: curtail PV real power output as local voltage rises.

    Args:
        circuit: An already-built circuit with at least one `PVSystem` element.
    """
    circuit.command("New XYCurve.vw_curve npts=4 Xarray=(0.5, 1.05, 1.10, 1.5) Yarray=(1.0, 1.0, 0.0, 0.0)")
    circuit.command(
        "New InvControl.feeder_vw mode=VOLTWATT voltage_curvex_ref=rated voltwatt_curve=vw_curve "
        "DeltaP_factor=0.1 voltagechangetolerance=0.0001 varchangetolerance=0.1 EventLog=no"
    )
    circuit.command("Set maxcontroliter=1000")


def apply_volt_var(circuit: Circuit) -> None:
    """Feeder-wide Volt-VAr inverter control: absorb/inject reactive power as local voltage rises/falls.

    Args:
        circuit: An already-built circuit with at least one `PVSystem` element.
    """
    circuit.command("New XYCurve.vv_curve npts=4 Xarray=(0.5, 0.95, 1.05, 1.5) Yarray=(1.0, 1.0, -1.0, -1.0)")
    circuit.command("New InvControl.feeder_vv mode=VOLTVAR voltage_curvex_ref=rated vvc_curve1=vv_curve EventLog=no")
    circuit.command("Set maxcontroliter=1000")


def add_storage(
    circuit: Circuit,
    *,
    bus1: str,
    name: str,
    kw_rated: float = 3.0,
    kwh_rated: float = 10.0,
    charge_step_range: tuple[int, int] = (20, 34),
    n_steps: int = 48,
) -> None:
    """Add a customer-sited storage unit that charges during a fixed midday window.

    Charging during `charge_step_range` (default steps 20-34, the same
    late-morning-to-mid-afternoon window PV output peaks in this book's own
    data) absorbs local PV export before it reaches the network, the real
    mechanism this lever is meant to test, not just a state flag. Outside
    that window the unit is idle. A real, scheduled dispatch, driven by the
    same `LoadShape` mechanism every other profile in this book uses, not a
    closed-loop controller, kept simple and legible on purpose.

    Args:
        circuit: An already-built circuit.
        bus1: The bus to connect the storage unit to, matching a real customer's own `Load.bus1`.
        name: A unique OpenDSS element name for this storage unit.
        kw_rated: Real power rating in kW.
        kwh_rated: Energy capacity in kWh.
        charge_step_range: Half-open `(start, end)` half-hour step range to charge during.
        n_steps: Number of half-hour steps in the daily dispatch shape (48 for a full day).
    """
    dispatch = np.zeros(n_steps)
    start, end = charge_step_range
    dispatch[start:end] = -kw_rated  # negative kW = charging, OpenDSS Storage sign convention
    circuit.command(
        f"New Storage.{name} bus1={bus1} phases=1 kv=0.23 kwrated={kw_rated} kwhrated={kwh_rated} "
        "kwhstored=0 %reserve=0 dispmode=follow pf=1"
    )
    circuit.command(f"New LoadShape.{name}_dispatch npts={n_steps} minterval=30 useactual=1 pmult={dispatch.tolist()}")
    circuit.command(f"edit Storage.{name} daily={name}_dispatch")
