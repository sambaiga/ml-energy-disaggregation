"""A pythonic `Circuit` handle on OpenDSSDirect.py.

OpenDSSDirect.py's own API is a direct mirror of the OpenDSS COM interface:
iterating lines means calling `Lines.First()` / `Lines.Next()` by hand and
reading each property off a shared "active element" pointer. `Circuit` hides
that dance behind ordinary Python iteration and small dataclasses, so a
notebook can write `for line in circuit.lines:` instead. Bulk numeric results
(voltages, currents, losses) stay in :mod:`ark.dss.results`, which this class
delegates to, since those are naturally tables, not element sequences.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import pandas as pd

from ark.dss import results as _results


@dataclass(frozen=True)
class Line:
    """One line/service-drop element."""

    name: str
    bus1: str
    bus2: str
    phases: int
    length: float


@dataclass(frozen=True)
class Load:
    """One load element."""

    name: str
    bus1: str
    kw: float
    kvar: float
    pf: float
    phases: int


@dataclass(frozen=True)
class Transformer:
    """One transformer element."""

    name: str
    kva: float
    num_windings: int


@dataclass(frozen=True)
class PVSystem:
    """One PV system element."""

    name: str
    bus1: str
    kva: float
    pmpp: float
    pf: float
    phases: int


@dataclass(frozen=True)
class StorageUnit:
    """One storage (battery) element."""

    name: str
    bus1: str
    kw_rated: float
    kwh_rated: float
    state: str
    phases: int


def _iterate(collection: Any, build) -> Iterator[Any]:
    """Walk an OpenDSSDirect `First()`/`Next()` collection, yielding `build()` per element."""
    if not collection.First():
        return
    while True:
        yield build()
        if not collection.Next():
            break


class Circuit:
    """A solved (or solvable) OpenDSS circuit, backed by OpenDSSDirect.py.

    OpenDSSDirect.py is a singleton engine (one circuit compiled at a time
    per process), so `Circuit` is a thin handle around that global state
    rather than an object owning its own isolated model. Loading a new
    circuit with :meth:`load` replaces whatever was compiled before.

    Examples:
        >>> circuit = Circuit.load("simple_lv/Master.dss")
        >>> circuit.converged
        True
        >>> [line.name for line in circuit.lines]
        ['a_b', 'b_l1', 'b_l2', 'b_l3']
        >>> circuit.bus_voltages().query("bus == 'b'")["vmag_pu"].round(3).tolist()
        [0.979, 0.979, 0.979]
    """

    def __init__(self) -> None:
        import opendssdirect as dss

        self._dss = dss

    @classmethod
    def load(cls, master_dss_path: str | Path, solve: bool = True) -> Circuit:
        """Clear the engine, compile `master_dss_path`, and optionally solve it.

        Args:
            master_dss_path: Path to the OpenDSS master `.dss` file (the one
                you would `Redirect` to). All `Redirect`-ed component files
                it references are followed automatically.
            solve: If True, runs a power flow solve immediately after
                compiling.

        Returns:
            A `Circuit` pointed at the compiled circuit.
        """
        circuit = cls()
        circuit._dss.Command("Clear")
        circuit._dss.Command(f'Redirect "{Path(master_dss_path)}"')
        if solve:
            circuit.solve()
        return circuit

    def command(self, text: str) -> str:
        """Run a raw OpenDSS command, for anything this class doesn't wrap.

        Returns the command's text result (e.g. the path `export` just
        wrote to, or a `?` query's value). `opendssdirect.Command()` itself
        always returns `None`; the result lives in `Text.Result()` after
        the command runs, so this reads it back for the caller.
        """
        self._dss.Command(text)
        return self._dss.Text.Result()

    @property
    def dss(self) -> ModuleType:
        """The raw `opendssdirect` module, for anything this class doesn't wrap.

        `Circuit` only models the handful of element types and result
        tables a book chapter has actually needed so far. Everything else
        OpenDSSDirect.py exposes (monitors, meters, PVSystems, per-terminal
        currents, ...) is still reachable here, unwrapped.
        """
        return self._dss

    def solve(self) -> None:
        """Run a power flow solve on the currently compiled circuit."""
        self._dss.Solution.Solve()

    @property
    def converged(self) -> bool:
        """Whether the last solve converged."""
        return self._dss.Solution.Converged()

    @property
    def bus_names(self) -> list[str]:
        """Every bus name in the circuit."""
        return self._dss.Circuit.AllBusNames()

    @property
    def source_bus(self) -> str:
        """The circuit's voltage-source (slack) bus name.

        A slack bus is held at its own fixed per-unit setpoint by
        construction (whatever `pu=` the `Vsource`/`Edit vsource.source`
        command set), not by anything the network does, so it reads as a
        real voltage everywhere `bus_voltages()` is used but says nothing
        about customer voltage. Any network-wide min/max over
        `bus_voltages()` should exclude it first
        (`voltages.query("bus != @circuit.source_bus")`), the same way a
        caller interested in one customer bus already filters `bus ==` to
        that name.
        """
        vsources = self._dss.Vsources
        vsources.First()
        self._dss.Circuit.SetActiveElement(f"Vsource.{vsources.Name()}")
        return self._dss.CktElement.BusNames()[0].split(".")[0]

    @property
    def lines(self) -> Iterator[Line]:
        """Iterate every line element in the circuit."""
        lines = self._dss.Lines
        yield from _iterate(
            lines,
            lambda: Line(
                name=lines.Name(),
                bus1=lines.Bus1(),
                bus2=lines.Bus2(),
                phases=lines.Phases(),
                length=lines.Length(),
            ),
        )

    @property
    def loads(self) -> Iterator[Load]:
        """Iterate every load element in the circuit."""
        loads = self._dss.Loads
        cktelement = self._dss.CktElement
        yield from _iterate(
            loads,
            lambda: Load(
                name=loads.Name(),
                bus1=cktelement.BusNames()[0],
                kw=loads.kW(),
                kvar=loads.kvar(),
                pf=loads.PF(),
                phases=loads.Phases(),
            ),
        )

    @property
    def transformers(self) -> Iterator[Transformer]:
        """Iterate every transformer element in the circuit."""
        transformers = self._dss.Transformers
        yield from _iterate(
            transformers,
            lambda: Transformer(
                name=transformers.Name(),
                kva=transformers.kVA(),
                num_windings=transformers.NumWindings(),
            ),
        )

    @property
    def pvsystems(self) -> Iterator[PVSystem]:
        """Iterate every PV system element in the circuit."""
        pvsystems = self._dss.PVsystems
        cktelement = self._dss.CktElement
        yield from _iterate(
            pvsystems,
            lambda: PVSystem(
                name=pvsystems.Name(),
                bus1=cktelement.BusNames()[0],
                kva=pvsystems.kVARated(),
                pmpp=pvsystems.Pmpp(),
                pf=pvsystems.pf(),
                phases=cktelement.NumPhases(),
            ),
        )

    @property
    def storage_units(self) -> Iterator[StorageUnit]:
        """Iterate every storage (battery) element in the circuit.

        `kw_rated`/`kwh_rated` aren't exposed on OpenDSSDirect.py's own
        `Storages` module (only `Name`/`State`/`puSOC` are), so this reads
        them the same escape-hatch way `circuit.command` documents
        elsewhere: a `?` property query against the active element.
        """
        storages = self._dss.Storages
        cktelement = self._dss.CktElement
        yield from _iterate(
            storages,
            lambda: StorageUnit(
                name=storages.Name(),
                bus1=cktelement.BusNames()[0],
                kw_rated=float(self.command(f"? Storage.{storages.Name()}.kwrated")),
                kwh_rated=float(self.command(f"? Storage.{storages.Name()}.kwhrated")),
                state=self.command(f"? Storage.{storages.Name()}.state"),
                phases=cktelement.NumPhases(),
            ),
        )

    def solve_daily(self, steps: int = 48, stepsize: str = "30m") -> Iterator[int]:
        """Step a `Set Mode=daily` time-series solve, yielding the step index after each solve.

        LoadShapes attached via `daily=` (see the Chapter 2 notebook) drive
        each load/PV system to a different point in its profile every step;
        a single `circuit.solve()` only ever answers "what does this network
        look like right now," this answers "what does it look like across a
        day." Read any per-step result (`circuit.bus_voltages()`,
        `circuit.element_powers(...)`) inside the loop, since each `Circuit`
        query only ever reflects the most recently solved step.

        Args:
            steps: Number of steps to solve (48 half-hour steps covers one
                full day at the default `stepsize`).
            stepsize: OpenDSS time-step duration string, e.g. `"30m"`, `"1h"`.

        Yields:
            The zero-based step index, after that step's solve has run.

        Examples:
            >>> circuit = Circuit.load("simple_lv/Master.dss", solve=False)
            >>> voltages = []
            >>> for step in circuit.solve_daily(steps=48):
            ...     voltages.append(circuit.bus_voltages().assign(step=step))
        """
        self._dss.Text.Command(f"Set Mode=daily number=1 stepsize={stepsize}")
        for step in range(steps):
            self.solve()
            yield step

    def bus_voltages(self) -> pd.DataFrame:
        """Per-phase bus voltage magnitude and angle for every bus. See :func:`ark.dss.results.bus_voltages`."""
        return _results.bus_voltages(self._dss)

    def line_currents(self) -> pd.DataFrame:
        """Per-phase current magnitude for every line. See :func:`ark.dss.results.line_currents`."""
        return _results.line_currents(self._dss)

    def line_losses(self) -> pd.DataFrame:
        """Active/reactive power loss for every line. See :func:`ark.dss.results.line_losses`."""
        return _results.line_losses(self._dss)

    def element_powers(self, element_type: str) -> pd.DataFrame:
        """Per-phase active/reactive power for every element of one type. See :func:`ark.dss.results.element_powers`."""
        return _results.element_powers(self._dss, element_type)

    def summary(self) -> dict:
        """A one-shot scalar health check for the circuit. See :func:`ark.dss.results.circuit_summary`."""
        return _results.circuit_summary(self._dss)
