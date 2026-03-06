#!/usr/bin/env python3
import argparse
import datetime as dt
import json
import os
import platform
import re
import shutil
import subprocess
from pathlib import Path


def _run(cmd):
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
        return out.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _parse_lscpu(text):
    if not text:
        return {}
    out = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, val = [x.strip() for x in line.split(":", 1)]
        out[key] = val
    return {
        "model": out.get("Model name") or out.get("Model"),
        "cpus": out.get("CPU(s)"),
        "threads_per_core": out.get("Thread(s) per core"),
        "cores_per_socket": out.get("Core(s) per socket"),
        "sockets": out.get("Socket(s)"),
        "numa_nodes": out.get("NUMA node(s)"),
        "max_mhz": out.get("CPU max MHz") or out.get("CPU MHz"),
    }


def _parse_meminfo(text):
    if not text:
        return {}
    out = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, val = [x.strip() for x in line.split(":", 1)]
        out[key] = val
    return {"mem_total_kb": out.get("MemTotal")}


def _parse_lsblk(text):
    if not text:
        return []
    disks = []
    for line in text.splitlines():
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 5:
            continue
        name, model, size, rota, tran = parts[:5]
        disks.append(
            {
                "name": name,
                "model": model or None,
                "size": size,
                "rotational": rota,
                "tran": tran or None,
            }
        )
    return disks


def _linux_summary():
    lscpu = _parse_lscpu(_run(["lscpu"]))
    meminfo = _parse_meminfo(_run(["cat", "/proc/meminfo"]))
    lsblk = _parse_lsblk(_run(["lsblk", "-d", "-o", "NAME,MODEL,SIZE,ROTA,TRAN", "--noheadings", "--pairs"]))
    # Fallback to pipe format if --pairs isn't supported
    if not lsblk:
        lsblk = _parse_lsblk(_run(["lsblk", "-d", "-o", "NAME,MODEL,SIZE,ROTA,TRAN", "--noheadings", "--output", "NAME,MODEL,SIZE,ROTA,TRAN", "--bytes", "--nodeps", "--noheadings"]))
    df_root = _run(["df", "-P", "/"])
    root_device = None
    if df_root:
        lines = df_root.splitlines()
        if len(lines) >= 2:
            root_device = lines[1].split()[0]
    numa = _run(["numactl", "--hardware"]) if shutil.which("numactl") else None
    gpus = _run(["nvidia-smi", "-L"]) if shutil.which("nvidia-smi") else None
    return {
        "cpu": lscpu,
        "memory": meminfo,
        "disks": lsblk,
        "root_device": root_device,
        "numa": numa,
        "gpus": gpus,
    }


def _mac_summary():
    cpu = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
    ncpu = _run(["sysctl", "-n", "hw.ncpu"])
    phys = _run(["sysctl", "-n", "hw.physicalcpu"])
    mem = _run(["sysctl", "-n", "hw.memsize"])
    hw = _run(["system_profiler", "SPHardwareDataType"])
    disk = _run(["diskutil", "info", "/"])
    return {
        "cpu": {
            "model": cpu,
            "cpus": ncpu,
            "physical_cores": phys,
        },
        "memory": {"mem_total_bytes": mem},
        "hardware": hw,
        "disk_info": disk,
    }


def _format_markdown(summary):
    lines = []
    lines.append(f"- Hostname: {summary.get('hostname')}")
    lines.append(f"- OS: {summary.get('os')} {summary.get('os_release')}")
    cpu = summary.get("cpu") or {}
    if cpu:
        lines.append(f"- CPU: {cpu.get('model')}")
        lines.append(f"- CPU(s): {cpu.get('cpus')} (threads/core {cpu.get('threads_per_core')}, cores/socket {cpu.get('cores_per_socket')}, sockets {cpu.get('sockets')})")
        lines.append(f"- CPU max MHz: {cpu.get('max_mhz')}")
    mem = summary.get("memory") or {}
    if mem:
        lines.append(f"- Mem total: {mem.get('mem_total_kb') or mem.get('mem_total_bytes')}")
    disks = summary.get("disks") or []
    if disks:
        lines.append("- Disks:")
        for d in disks:
            lines.append(f"  - {d.get('name')}: {d.get('model')} {d.get('size')} (rotational {d.get('rotational')}, tran {d.get('tran')})")
    if summary.get("gpus"):
        lines.append("- GPUs:")
        lines.extend([f"  - {line}" for line in summary["gpus"].splitlines()])
    return "\n".join(lines)


def _format_markdown_simple(summary):
    lines = []
    lines.append(f"- Hostname: {summary.get('hostname')}")
    lines.append(f"- OS: {summary.get('os')} {summary.get('os_release')}")
    cpu = summary.get("cpu") or {}
    if cpu:
        lines.append(f"- CPU: {cpu.get('model')}")
        lines.append(f"- CPU(s): {cpu.get('cpus')}")
    mem = summary.get("memory") or {}
    if mem:
        lines.append(f"- Mem total: {mem.get('mem_total_kb') or mem.get('mem_total_bytes')}")
    disks = summary.get("disks") or []
    if disks:
        rota = [d.get('rotational') for d in disks if d.get('rotational') is not None]
        lines.append(f"- Disks: {len(disks)} (rotational: {','.join(rota) if rota else 'unknown'})")
    if summary.get("gpus"):
        lines.append("- GPUs: present")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Collect a machine summary for experiment logs.")
    parser.add_argument("--out", default=None, help="Output JSON path.")
    parser.add_argument("--append-csv", default=None, help="Append a single-line CSV summary.")
    parser.add_argument("--tag", default=None, help="Optional experiment tag.")
    parser.add_argument("--simple", action="store_true", help="Write a short hardware-only summary.")
    args = parser.parse_args()

    system = platform.system()
    base = {
        "timestamp": dt.datetime.utcnow().isoformat() + "Z",
        "hostname": platform.node(),
        "os": system,
        "os_release": platform.release(),
        "os_version": platform.version(),
        "tag": args.tag,
    }

    if system == "Linux":
        base.update(_linux_summary())
    elif system == "Darwin":
        base.update(_mac_summary())
    else:
        base["note"] = f"Unsupported OS: {system}"

    out_path = args.out
    if out_path is None:
        out_dir = Path("logs") / "machine_summaries"
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        out_path = out_dir / f"machine_summary_{base['hostname']}_{stamp}.json"
    out_path = Path(out_path)
    out_path.write_text(json.dumps(base, indent=2))

    if args.append_csv:
        csv_path = Path(args.append_csv)
        header = [
            "timestamp",
            "hostname",
            "os",
            "os_release",
            "cpu_model",
            "cpus",
            "threads_per_core",
            "cores_per_socket",
            "sockets",
            "mem_total",
            "tag",
        ]
        cpu = base.get("cpu") or {}
        mem = base.get("memory") or {}
        row = [
            base.get("timestamp"),
            base.get("hostname"),
            base.get("os"),
            base.get("os_release"),
            cpu.get("model"),
            cpu.get("cpus"),
            cpu.get("threads_per_core"),
            cpu.get("cores_per_socket"),
            cpu.get("sockets"),
            mem.get("mem_total_kb") or mem.get("mem_total_bytes"),
            base.get("tag"),
        ]
        if not csv_path.exists():
            csv_path.write_text(",".join(header) + "\n")
        with csv_path.open("a") as f:
            f.write(",".join("" if v is None else str(v) for v in row) + "\n")

    md_path = out_path.with_suffix(".md")
    md_path.write_text(_format_markdown_simple(base) if args.simple else _format_markdown(base))

    print(f"Wrote {out_path}")
    print(f"Wrote {md_path}")
    if args.append_csv:
        print(f"Appended {args.append_csv}")


if __name__ == "__main__":
    main()
