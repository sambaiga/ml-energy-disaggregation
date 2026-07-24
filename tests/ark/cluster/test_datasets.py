import io
import tarfile
import zipfile

import numpy as np
import pandas as pd
import pytest

import ark.cluster.datasets as datasets_module
from ark.cluster.datasets import (
    load_ausnet_pivot,
    load_ausnet_shape,
    load_goiener_pivot,
    load_goiener_shape,
    load_london_pivot,
    load_london_shape,
)


def test_load_ausnet_shape_missing_file_raises_system_exit(tmp_path, monkeypatch):
    monkeypatch.setattr(datasets_module, "_RESOURCES", tmp_path)

    with pytest.raises(SystemExit, match="fetch_part4_ausnet_data"):
        load_ausnet_shape()


def test_load_ausnet_shape_returns_peak_normalised_whole_year_mean(tmp_path, monkeypatch):
    monkeypatch.setattr(datasets_module, "_RESOURCES", tmp_path)
    data_dir = tmp_path / "cvar_flexibility" / "data" / "timeseries-lv"
    data_dir.mkdir(parents=True)

    # 3 households, 360 real days, 4 half-hours/day; household 0 has a real,
    # checkable known peak (10.0) so the normalisation itself is verifiable.
    rng = np.random.default_rng(0)
    load_data = rng.uniform(0, 5, size=(3, 360, 4))
    load_data[0, 0, 0] = 10.0
    np.save(data_dir / "Residential load data 30-min resolution.npy", load_data)

    shape = load_ausnet_shape()

    assert shape.shape == (3, 4)
    assert np.isclose(shape.to_numpy().max(axis=1), 1.0).all()  # every household peaks at exactly 1.0
    assert list(shape.index) == [0, 1, 2]


def test_load_ausnet_pivot_returns_a_real_datetime_index_and_float_dtype(tmp_path, monkeypatch):
    monkeypatch.setattr(datasets_module, "_RESOURCES", tmp_path)
    data_dir = tmp_path / "cvar_flexibility" / "data" / "timeseries-lv"
    data_dir.mkdir(parents=True)

    rng = np.random.default_rng(0)
    # Integer-valued readings (a real edge case: some real households' whole
    # window happens to round to whole numbers, which numpy/pandas would
    # otherwise infer as int64, failing extract_features's own float64
    # contract downstream).
    load_data = rng.integers(0, 5, size=(2, 360, 4)).astype(np.float64)
    np.save(data_dir / "Residential load data 30-min resolution.npy", load_data)

    pivot = load_ausnet_pivot()

    assert pivot.shape == (360 * 4, 2)
    assert pd.api.types.is_datetime64_any_dtype(pivot.index)
    assert (pivot.dtypes == np.float64).all()


def _write_london_archive(path, rng, n_households=3, n_days=360):
    idx = pd.date_range("2013-01-01", periods=n_days * 48, freq="30min")
    with zipfile.ZipFile(path, "w") as z:
        rows = []
        for h in range(n_households):
            values = rng.uniform(0, 2, size=len(idx))
            df = pd.DataFrame(
                {
                    "LCLid": f"MAC{h:06d}",
                    "DateTime": idx,
                    "KWH/hh (per half hour)": values,
                }
            )
            rows.append(df)
        combined = pd.concat(rows, ignore_index=True)
        z.writestr("block_0.csv", combined.to_csv(index=False))


def test_load_london_shape_returns_peak_normalised_whole_year_mean(tmp_path, monkeypatch):
    monkeypatch.setattr(datasets_module, "_RESOURCES", tmp_path)
    data_dir = tmp_path / "london-lcl" / "data"
    data_dir.mkdir(parents=True)
    rng = np.random.default_rng(1)
    _write_london_archive(data_dir / "Partitioned LCL Data.zip", rng)

    shape = load_london_shape(n_partitions=1, min_coverage=0.5)

    assert shape.shape == (3, 48)
    assert np.isclose(shape.to_numpy().max(axis=1), 1.0).all()


def test_load_london_pivot_returns_float_dtype_even_from_integer_readings(tmp_path, monkeypatch):
    monkeypatch.setattr(datasets_module, "_RESOURCES", tmp_path)
    data_dir = tmp_path / "london-lcl" / "data"
    data_dir.mkdir(parents=True)

    class _IntegerRNG:
        def uniform(self, low, high, size):
            return np.zeros(size)  # every reading is exactly 0.0, a real int-like edge case

    _write_london_archive(data_dir / "Partitioned LCL Data.zip", _IntegerRNG())

    pivot = load_london_pivot(n_partitions=1, min_coverage=0.5)

    assert (pivot.dtypes == np.float64).all()
    assert pd.api.types.is_datetime64_any_dtype(pivot.index)


def _write_goiener_archive(archive_path, metadata_path, rng, n_households=3, n_days=360):
    import zstandard as zstd

    start = pd.Timestamp("2021-06-06", tz="UTC")
    end = start + pd.Timedelta(days=n_days)
    idx = pd.date_range("2021-06-06", periods=n_days * 24, freq="1h")

    meta_rows = []
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for h in range(n_households):
            user_id = f"user{h}"
            values = rng.uniform(0, 3, size=len(idx))
            df = pd.DataFrame({"timestamp": idx, "kWh": values})
            csv_bytes = df.to_csv(index=False).encode()
            info = tarfile.TarInfo(name=f"{user_id}.csv")
            info.size = len(csv_bytes)
            tar.addfile(info, io.BytesIO(csv_bytes))
            meta_rows.append(
                {
                    "user": user_id,
                    "cnae": 9820.0,
                    "missing_samples_pct": 0.0,
                    "length_years": 1.5,
                    "start_date": start,
                    "end_date": end,
                }
            )
    dctx = zstd.ZstdCompressor()
    archive_path.write_bytes(dctx.compress(buf.getvalue()))
    pd.DataFrame(meta_rows).to_csv(metadata_path, index=False)


def test_load_goiener_shape_returns_peak_normalised_whole_year_mean(tmp_path, monkeypatch):
    monkeypatch.setattr(datasets_module, "_RESOURCES", tmp_path)
    data_dir = tmp_path / "goiener" / "data"
    data_dir.mkdir(parents=True)
    rng = np.random.default_rng(2)
    _write_goiener_archive(data_dir / "imp-post.tzst", data_dir / "metadata.csv", rng)

    shape = load_goiener_shape(n_households=3, min_coverage=0.5)

    assert shape.shape == (3, 24)
    assert np.isclose(shape.to_numpy().max(axis=1), 1.0).all()


def test_load_goiener_pivot_returns_float_dtype_even_from_integer_readings(tmp_path, monkeypatch):
    monkeypatch.setattr(datasets_module, "_RESOURCES", tmp_path)
    data_dir = tmp_path / "goiener" / "data"
    data_dir.mkdir(parents=True)

    class _ZeroRNG:
        def uniform(self, low, high, size):
            return np.zeros(size)  # every reading exactly 0.0, the real int64-inference edge case

    _write_goiener_archive(data_dir / "imp-post.tzst", data_dir / "metadata.csv", _ZeroRNG())

    pivot = load_goiener_pivot(n_households=3, min_coverage=0.5)

    assert (pivot.dtypes == np.float64).all()
    assert pd.api.types.is_datetime64_any_dtype(pivot.index)


def test_load_ausnet_pivot_missing_file_raises_system_exit(tmp_path, monkeypatch):
    monkeypatch.setattr(datasets_module, "_RESOURCES", tmp_path)

    with pytest.raises(SystemExit, match="fetch_part4_ausnet_data"):
        load_ausnet_pivot()


def test_load_london_pivot_missing_file_raises_system_exit(tmp_path, monkeypatch):
    monkeypatch.setattr(datasets_module, "_RESOURCES", tmp_path)

    with pytest.raises(SystemExit, match="fetch_london_lcl_data"):
        load_london_pivot()


def test_load_goiener_pivot_missing_file_raises_system_exit(tmp_path, monkeypatch):
    monkeypatch.setattr(datasets_module, "_RESOURCES", tmp_path)

    with pytest.raises(SystemExit, match="fetch_goiener_data"):
        load_goiener_pivot()


def test_load_goiener_pivot_skips_non_file_and_non_target_tar_members(tmp_path, monkeypatch):
    # A real GoiEner archive holds a directory entry (isfile() is False) and
    # more households than any one sample targets; both real streaming-loop
    # branches (skip non-file members, skip members outside the stratified
    # sample) need a household pool larger than the sample to ever fire.
    monkeypatch.setattr(datasets_module, "_RESOURCES", tmp_path)
    data_dir = tmp_path / "goiener" / "data"
    data_dir.mkdir(parents=True)
    rng = np.random.default_rng(3)
    _write_goiener_archive(data_dir / "imp-post.tzst", data_dir / "metadata.csv", rng, n_households=5)

    import zstandard as zstd

    archive_path = data_dir / "imp-post.tzst"
    dctx = zstd.ZstdDecompressor()
    with archive_path.open("rb") as fh:
        raw_tar = dctx.decompress(fh.read())
    buf = io.BytesIO(raw_tar)
    with tarfile.open(fileobj=buf, mode="r") as tar:
        members = {m.name: tar.extractfile(m).read() for m in tar.getmembers()}
    new_buf = io.BytesIO()
    with tarfile.open(fileobj=new_buf, mode="w") as tar:
        dir_info = tarfile.TarInfo(name="subdir")
        dir_info.type = tarfile.DIRTYPE
        tar.addfile(dir_info)
        for name, content in members.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(content)
            tar.addfile(info, io.BytesIO(content))
    archive_path.write_bytes(zstd.ZstdCompressor().compress(new_buf.getvalue()))

    pivot = load_goiener_pivot(n_households=3, min_coverage=0.5)

    assert pivot.shape[1] <= 3  # sampled a real subset, not all 5 real households on disk
