#!/usr/bin/env python3
"""Summarize and plot CPU/CUDA direct-beamformer benchmark CSV files."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np


TIME_FIELDS = ("cpu_ms", "gpu_kernel_ms", "gpu_pipeline_wall_ms")
LOCAL_FREQUENCY_CHANNELS = 336
FULL_BAND_FREQUENCY_CHANNELS = 672


def with_suffix(prefix: Path, suffix: str) -> Path:
    return prefix.parent / f"{prefix.name}{suffix}"


def load_numeric_csv(path: Path) -> list[dict[str, float]]:
    with path.open(newline="") as input_file:
        reader = csv.DictReader(input_file)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no CSV header")
        records = [
            {name: float(value) for name, value in row.items()}
            for row in reader
        ]
    if not records:
        raise ValueError(f"{path} contains no data rows")
    return records


def filter_records_for_buffer(records: Iterable[dict[str, float]],
                              buffer: str | None) -> list[dict[str, float]]:
    records = list(records)
    available = {int(record["n_freq"]) for record in records}
    if buffer is None:
        if len(available) > 1:
            raise ValueError("multiple n_freq values found; select --buffer")
        return records
    expected = FULL_BAND_FREQUENCY_CHANNELS if buffer == "both" else LOCAL_FREQUENCY_CHANNELS
    selected = [record for record in records if int(record["n_freq"]) == expected]
    if not selected:
        raise ValueError(f"no benchmark records for buffer={buffer} (n_freq={expected})")
    return selected


def summarize_timings(records: Iterable[dict[str, float]]) -> list[dict[str, float]]:
    groups: dict[tuple[int, int, int, int], list[dict[str, float]]] = defaultdict(list)
    for record in records:
        key = (int(record["n_ant"]), int(record["n_freq"]),
               int(record["n_beams"]), int(record["n_time"]))
        groups[key].append(record)

    summary: list[dict[str, float]] = []
    for (n_ant, n_freq, n_beams, n_time), rows in sorted(groups.items()):
        first = rows[0]
        result: dict[str, float] = {
            "n_ant": float(n_ant),
            "n_freq": float(n_freq),
            "n_beams": float(n_beams),
            "n_time": float(n_time),
            "n_outputs": first["n_outputs"],
            "n_cmac": first["n_cmac"],
            "estimated_flop": first["estimated_flop"],
            "repetitions": float(len(rows)),
        }
        available_time_fields = [field for field in TIME_FIELDS if field in first]
        for field in available_time_fields:
            values = np.asarray([row[field] for row in rows], dtype=np.float64)
            result[f"{field}_median"] = float(np.median(values))
            result[f"{field}_p25"] = float(np.percentile(values, 25.0))
            result[f"{field}_p75"] = float(np.percentile(values, 75.0))

        kernel_ms = result["gpu_kernel_ms_median"]
        pipeline_ms = result["gpu_pipeline_wall_ms_median"]
        if "cpu_ms_median" in result:
            cpu_ms = result["cpu_ms_median"]
            result["speedup_kernel"] = cpu_ms / kernel_ms
            result["speedup_pipeline"] = cpu_ms / pipeline_ms
        rate_values = [
            ("gpu_kernel", kernel_ms),
            ("gpu_pipeline", pipeline_ms),
        ]
        if "cpu_ms_median" in result:
            rate_values.insert(0, ("cpu", result["cpu_ms_median"]))
        for label, milliseconds in rate_values:
            seconds = milliseconds / 1000.0
            result[f"{label}_cmac_per_s"] = result["n_cmac"] / seconds
            result[f"{label}_estimated_flop_per_s"] = (
                result["estimated_flop"] / seconds
            )
        summary.append(result)
    return summary


def write_summary(path: Path, records: list[dict[str, float]]) -> None:
    if not records:
        raise ValueError("cannot write an empty summary")
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def matrix_for(records: Iterable[dict[str, float]], n_ant: int,
               value_field: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected = [record for record in records if int(record["n_ant"]) == n_ant]
    if not selected:
        raise ValueError(f"no records for n_ant={n_ant}")
    beams = np.asarray(sorted({int(record["n_beams"]) for record in selected}))
    times = np.asarray(sorted({int(record["n_time"]) for record in selected}))
    matrix = np.full((len(beams), len(times)), np.nan, dtype=np.float64)
    beam_index = {value: index for index, value in enumerate(beams)}
    time_index = {value: index for index, value in enumerate(times)}
    for record in selected:
        matrix[beam_index[int(record["n_beams"])],
               time_index[int(record["n_time"])]] = record[value_field]
    return beams, times, matrix


def positive_log_limits(values: np.ndarray) -> tuple[float, float] | None:
    values = np.asarray(values, dtype=np.float64)
    positive = values[np.isfinite(values) & (values > 0.0)]
    if not positive.size:
        return None
    lower = float(np.min(positive))
    upper = float(np.max(positive))
    if lower == upper:
        lower = upper / 10.0
    return lower, upper


def _records_for(summary: list[dict[str, float]], n_ant: int,
                 n_beams: int) -> list[dict[str, float]]:
    return sorted(
        [record for record in summary
         if int(record["n_ant"]) == n_ant
         and int(record["n_beams"]) == n_beams],
        key=lambda record: record["n_time"],
    )


def plot_performance(summary: list[dict[str, float]], metadata: dict,
                     output_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    antenna_values = sorted({int(record["n_ant"]) for record in summary})
    has_cpu_timings = "cpu_ms_median" in summary[0]
    figure, axes = plt.subplots(
        len(antenna_values), 3, figsize=(18, 5.2 * len(antenna_values)),
        squeeze=False, constrained_layout=True,
    )
    for row, n_ant in enumerate(antenna_values):
        beam_values = sorted({int(record["n_beams"]) for record in summary
                              if int(record["n_ant"]) == n_ant})
        max_beams = max(beam_values)
        maximum = _records_for(summary, n_ant, max_beams)
        times = np.asarray([record["n_time"] for record in maximum])

        timing_axis = axes[row, 0]
        timing_fields = [
            ("gpu_kernel_ms", "CUDA kernel", "tab:orange"),
            ("gpu_pipeline_wall_ms", "GPU H2D+kernel+D2H", "tab:green"),
        ]
        if has_cpu_timings:
            timing_fields.insert(0, ("cpu_ms", "CPU serial", "tab:blue"))
        for field, label, color in timing_fields:
            median = np.asarray([record[f"{field}_median"] for record in maximum])
            p25 = np.asarray([record[f"{field}_p25"] for record in maximum])
            p75 = np.asarray([record[f"{field}_p75"] for record in maximum])
            timing_axis.plot(times, median, marker="o", label=label, color=color)
            timing_axis.fill_between(times, p25, p75, color=color, alpha=0.15)
        timing_axis.set_xscale("log")
        timing_axis.set_yscale("log")
        timing_axis.set_xlabel("n_time")
        timing_axis.set_ylabel("Tiempo [ms]")
        timing_axis.set_title(f"A={n_ant}, B={max_beams}: mediana e IQR")
        timing_axis.grid(True, which="both", alpha=0.3)
        timing_axis.legend()

        comparison_axis = axes[row, 1]
        for n_beams in beam_values:
            beam_records = _records_for(summary, n_ant, n_beams)
            field = "speedup_pipeline" if has_cpu_timings \
                else "gpu_pipeline_wall_ms_median"
            comparison_axis.plot(
                [record["n_time"] for record in beam_records],
                [record[field] for record in beam_records],
                marker="o", label=f"B={n_beams}",
            )
        if has_cpu_timings:
            comparison_axis.axhline(
                1.0, color="black", linestyle="--", linewidth=1.2,
                label="CPU = GPU")
            comparison_axis.set_ylabel("Speedup CPU / pipeline GPU")
            comparison_axis.set_title(f"Frontera de conveniencia, A={n_ant}")
        else:
            comparison_axis.set_ylabel("Pipeline GPU [ms]")
            comparison_axis.set_title(f"Escalamiento GPU por beams, A={n_ant}")
        comparison_axis.set_xscale("log")
        comparison_axis.set_yscale("log")
        comparison_axis.set_xlabel("n_time")
        comparison_axis.grid(True, which="both", alpha=0.3)
        comparison_axis.legend(ncol=2, fontsize=9)

        throughput_axis = axes[row, 2]
        throughput_fields = [
            ("gpu_kernel_estimated_flop_per_s", "CUDA kernel", "tab:orange"),
            ("gpu_pipeline_estimated_flop_per_s", "GPU pipeline", "tab:green"),
        ]
        if has_cpu_timings:
            throughput_fields.insert(
                0, ("cpu_estimated_flop_per_s", "CPU serial", "tab:blue"))
        for field, label, color in throughput_fields:
            throughput_axis.plot(
                [record["n_cmac"] for record in maximum],
                [record[field] / 1.0e9 for record in maximum],
                marker="o", label=label, color=color,
            )
        throughput_axis.set_xscale("log")
        throughput_axis.set_yscale("log")
        throughput_axis.set_xlabel("Carga Ncmac")
        throughput_axis.set_ylabel("Throughput estimado [GFLOP/s]")
        throughput_axis.set_title(f"A={n_ant}, B={max_beams}")
        throughput_axis.grid(True, which="both", alpha=0.3)
        throughput_axis.legend()

    gpu_name = metadata.get("gpu_name", "GPU desconocida")
    repetitions = metadata.get("repetitions", "?")
    n_freq = int(summary[0]["n_freq"])
    figure.suptitle(
        f"Beamformer {'CPU/CUDA' if has_cpu_timings else 'CUDA'} — "
        f"{gpu_name}; F={n_freq} channels; {repetitions} repeticiones\n"
        "Ncmac=T×F×B×A; FLOP estimados=8×Ncmac+3×Noutput",
        fontsize=14,
    )
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def _annotate_matrix(axis, matrix: np.ndarray, formatter: str,
                     norm=None, colormap=None) -> None:
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            if np.isfinite(value):
                text_color = "black"
                if norm is not None and colormap is not None and value > 0.0:
                    red, green, blue, _ = colormap(norm(value))
                    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
                    text_color = "white" if luminance < 0.48 else "black"
                axis.text(column, row, format(value, formatter), ha="center",
                          va="center", fontsize=8, color=text_color)


def plot_speedup_heatmaps(summary: list[dict[str, float]],
                          output_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    antenna_values = sorted({int(record["n_ant"]) for record in summary})
    figure, axes = plt.subplots(
        len(antenna_values), 2, figsize=(14, 4.8 * len(antenna_values)),
        squeeze=False, constrained_layout=True,
    )
    for row, n_ant in enumerate(antenna_values):
        for column, (field, title) in enumerate((
            ("speedup_kernel", "CPU / CUDA kernel"),
            ("speedup_pipeline", "CPU / GPU pipeline"),
        )):
            beams, times, matrix = matrix_for(summary, n_ant, field)
            finite = matrix[np.isfinite(matrix)]
            positive = finite[finite > 0.0]
            lower = max(float(np.min(positive)), 0.05)
            upper = max(float(np.max(positive)), lower * 1.01)
            axis = axes[row, column]
            image = axis.imshow(matrix, aspect="auto", origin="lower", cmap="RdYlGn",
                                norm=LogNorm(vmin=lower, vmax=upper))
            if float(np.nanmin(matrix)) <= 1.0 <= float(np.nanmax(matrix)):
                axis.contour(matrix, levels=[1.0], colors="black", linewidths=2.0)
            axis.set_xticks(np.arange(len(times)), [str(value) for value in times],
                            rotation=35, ha="right")
            axis.set_yticks(np.arange(len(beams)), [str(value) for value in beams])
            axis.set_xlabel("n_time")
            axis.set_ylabel("n_beams")
            axis.set_title(f"{title}, n_ant={n_ant}")
            _annotate_matrix(axis, matrix, ".2g")
            figure.colorbar(image, ax=axis, label="Speedup")
    figure.suptitle("Speedup; el contorno negro marca speedup = 1", fontsize=14)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def plot_gpu_time_heatmaps(summary: list[dict[str, float]],
                           output_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    antenna_values = sorted({int(record["n_ant"]) for record in summary})
    figure, axes = plt.subplots(
        len(antenna_values), 2, figsize=(14, 4.8 * len(antenna_values)),
        squeeze=False, constrained_layout=True,
    )
    for row, n_ant in enumerate(antenna_values):
        for column, (field, title) in enumerate((
            ("gpu_kernel_ms_median", "CUDA kernel [ms]"),
            ("gpu_pipeline_wall_ms_median", "GPU pipeline [ms]"),
        )):
            beams, times, matrix = matrix_for(summary, n_ant, field)
            limits = positive_log_limits(matrix)
            if limits is None:
                raise ValueError(f"no positive GPU timings for {field}, n_ant={n_ant}")
            lower, upper = limits
            upper = max(upper, lower * 1.01)
            axis = axes[row, column]
            image = axis.imshow(
                matrix, aspect="auto", origin="lower", cmap="viridis",
                norm=LogNorm(vmin=lower, vmax=upper),
            )
            axis.set_xticks(np.arange(len(times)), [str(value) for value in times],
                            rotation=35, ha="right")
            axis.set_yticks(np.arange(len(beams)), [str(value) for value in beams])
            axis.set_xlabel("n_time")
            axis.set_ylabel("n_beams")
            axis.set_title(f"{title}, n_ant={n_ant}")
            _annotate_matrix(axis, matrix, ".3g")
            figure.colorbar(image, ax=axis, label="Tiempo [ms] (escala log)")
    figure.suptitle("Tiempos GPU; CPU se usa sólo para validación compacta", fontsize=14)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def plot_validation(validation: list[dict[str, float]],
                    output_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, LogNorm

    records = [dict(record, outside_fraction=(
        record["outside_tolerance"] / record["n_outputs"]))
        for record in validation]
    antenna_values = sorted({int(record["n_ant"]) for record in records})
    figure, axes = plt.subplots(
        len(antenna_values), 2, figsize=(14, 4.8 * len(antenna_values)),
        squeeze=False, constrained_layout=True,
    )
    for row, n_ant in enumerate(antenna_values):
        for column, (field, title) in enumerate((
            ("max_relative_error", "Error relativo máximo"),
            ("outside_fraction", "Fracción fuera de tolerancia"),
        )):
            beams, times, matrix = matrix_for(records, n_ant, field)
            axis = axes[row, column]
            log_limits = positive_log_limits(matrix)
            if log_limits is None:
                colormap = ListedColormap(["#e8f5e9"])
                axis.imshow(
                    np.zeros_like(matrix), aspect="auto", origin="lower",
                    cmap=colormap, vmin=0.0, vmax=1.0,
                )
                title_suffix = " | todas las configuraciones = 0"
                _annotate_matrix(axis, matrix, ".1e")
            else:
                colormap = plt.get_cmap("viridis").copy()
                colormap.set_bad("white")
                norm = LogNorm(vmin=log_limits[0], vmax=log_limits[1])
                display = np.ma.masked_invalid(np.ma.masked_less_equal(matrix, 0.0))
                image = axis.imshow(
                    display, aspect="auto", origin="lower", cmap=colormap, norm=norm,
                )
                title_suffix = " | log; blanco = 0"
                _annotate_matrix(axis, matrix, ".1e", norm, colormap)
                figure.colorbar(image, ax=axis, label="valor positivo (escala log)")
            axis.set_xticks(np.arange(len(times)), [str(value) for value in times],
                            rotation=35, ha="right")
            axis.set_yticks(np.arange(len(beams)), [str(value) for value in beams])
            axis.set_xlabel("n_time")
            axis.set_ylabel("n_beams")
            axis.set_title(f"{title}, n_ant={n_ant}{title_suffix}")
    figure.suptitle(
        "Validación numérica CPU vs CUDA | colores: valores positivos; "
        "blanco/verde: cero",
        fontsize=14,
    )
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-prefix", type=Path,
                        default=Path("results/gpu_benchmark_fft"))
    parser.add_argument("--output-prefix", type=Path)
    parser.add_argument("--buffer", choices=("0", "1", "both"),
                        help="filter local-buffer or full-band benchmark rows")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_prefix: Path = args.input_prefix
    output_prefix: Path = args.output_prefix or input_prefix
    timings_path = with_suffix(input_prefix, "_timings.csv")
    validation_path = with_suffix(input_prefix, "_validation.csv")
    metadata_path = with_suffix(input_prefix, "_metadata.json")
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    timings = load_numeric_csv(timings_path)
    validation = load_numeric_csv(validation_path)
    metadata = json.loads(metadata_path.read_text())
    timings = filter_records_for_buffer(timings, args.buffer)
    validation = filter_records_for_buffer(validation, args.buffer)
    summary = summarize_timings(timings)

    summary_path = with_suffix(output_prefix, "_summary.csv")
    performance_path = with_suffix(output_prefix, "_performance.png")
    has_cpu_timings = "cpu_ms_median" in summary[0]
    heatmap_path = with_suffix(
        output_prefix,
        "_speedup_heatmaps.png" if has_cpu_timings else "_gpu_time_heatmaps.png",
    )
    validation_output_path = with_suffix(output_prefix, "_validation.png")
    write_summary(summary_path, summary)
    plot_performance(summary, metadata, performance_path)
    if has_cpu_timings:
        plot_speedup_heatmaps(summary, heatmap_path)
    else:
        plot_gpu_time_heatmaps(summary, heatmap_path)
    plot_validation(validation, validation_output_path)
    print(f"Wrote {summary_path}")
    print(f"Wrote {performance_path}")
    print(f"Wrote {heatmap_path}")
    print(f"Wrote {validation_output_path}")


if __name__ == "__main__":
    main()
