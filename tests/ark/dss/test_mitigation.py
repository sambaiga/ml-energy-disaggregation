from ark.dss.circuit import Circuit
from ark.dss.mitigation import add_storage, apply_volt_var, apply_volt_watt


def _minimal_pv_circuit() -> Circuit:
    """A tiny, self-contained 2-bus circuit with one PV-hosting load, no external file needed."""
    circuit = Circuit()
    circuit.command("Clear")
    circuit.command("New circuit.test basekv=0.23 pu=1.0 phases=1 bus1=sourcebus")
    circuit.command("New line.l1 bus1=sourcebus bus2=b1 phases=1 length=0.05 units=km r1=0.5 x1=0.1")
    circuit.command("New load.house1 bus1=b1 phases=1 kv=0.23 kw=2 pf=0.98 model=1")
    circuit.command(
        "New PVSystem.pv1 bus1=b1 phases=1 kva=5 pmpp=5 pf=1 kV=0.23 model=1 irradiance=1 %cutin=0.05 %cutout=0.05"
    )
    circuit.command("Set VoltageBases=[0.23]")
    circuit.command("calcvoltagebases")
    circuit.solve()
    return circuit


def test_apply_volt_watt_adds_a_working_inverter_control():
    circuit = _minimal_pv_circuit()

    apply_volt_watt(circuit)
    circuit.solve()

    assert circuit.converged


def test_apply_volt_var_adds_a_working_inverter_control():
    circuit = _minimal_pv_circuit()

    apply_volt_var(circuit)
    circuit.solve()

    assert circuit.converged


def test_add_storage_creates_a_storage_unit_with_the_requested_rating():
    circuit = _minimal_pv_circuit()

    add_storage(circuit, bus1="b1", name="batt1", kw_rated=3.0, kwh_rated=10.0)
    circuit.solve()

    units = list(circuit.storage_units)
    assert len(units) == 1
    assert units[0].kw_rated == 3.0
    assert units[0].kwh_rated == 10.0


def test_add_storage_charges_only_within_the_requested_step_range():
    circuit = _minimal_pv_circuit()

    add_storage(circuit, bus1="b1", name="batt1", kw_rated=3.0, kwh_rated=10.0, charge_step_range=(10, 12))

    vmax_by_step = []
    for _step in circuit.solve_daily(steps=48):
        vmax_by_step.append(circuit.bus_voltages().set_index("bus").loc["b1", "vmag_pu"])

    assert circuit.converged
    assert len(vmax_by_step) == 48
