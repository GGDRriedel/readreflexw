import os
import re
from pathlib import Path
import numpy as np
import pandas as pd




# ---------- Helpers ----------

def _clean_value(s: str):
    if s is None:
        return None
    s = s.strip().strip("'").strip('"')
    return s if s != "" else None

def _try_number(s: str):
    """Try int, else float, else string."""
    try:
        if re.fullmatch(r"[+-]?\d+", s):
            return int(s)
        # Convert decimal with possible scientific notation
        return float(s)
    except Exception:
        return s

def _parse_list(s: str):
    """Parse a space-separated list of numbers into a numpy array."""
    s = s.strip()
    if not s:
        return np.array([])
    parts = [p for p in s.split() if p]
    # try numeric
    nums = []
    for p in parts:
        v = _try_number(p)
        nums.append(v if isinstance(v, (int, float)) else np.nan)
    # if any non-numeric slipped in, just return raw strings
    if any(np.isnan(x) for x in nums if isinstance(x, float)):
        return parts
    return np.array(nums, dtype=float)

def read_positions(pos_file: str | Path) -> pd.DataFrame:
    """
    Read a MALÅ position/coordinate file (.cor/.pos).
    Skips metadata lines like '*' and 'UNITS:m'.
    Returns a DataFrame with one row per trace.
    """
    pos_file = Path(pos_file)

    # First pass: read lines and skip until we hit numeric rows
    rows = []
    with pos_file.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.upper().startswith("UNITS"):
                continue
            rows.append(line)

    # Now parse the numeric part
    from io import StringIO
    buf = StringIO("\n".join(rows))

    try:
        df = pd.read_csv(buf, sep=None, engine="python", comment="#",header=None,names=["Trace","Y","X","Z"])
    except Exception:
        buf.seek(0)
        df = pd.read_csv(buf,sep='\s+', comment="#",header=None,names=["Trace","Y","X","Z"])

    # Normalize column names
    df.columns = [c.strip().upper() for c in df.columns]
    return df
def expand_positions_for_channels(pos_df: pd.DataFrame, n_channels: int):
    """
    Repeat each trace position across all channels.
    Returns a DataFrame with columns: TRACE, CHANNEL, plus the original pos columns.
    """
    n_traces = len(pos_df)
    traces = np.arange(n_traces)

    # Repeat trace indices and channels
    trace_idx = np.repeat(traces, n_channels)
    chan_idx = np.tile(np.arange(n_channels), n_traces)

    # Repeat position data
    pos_repeated = pd.DataFrame(
        np.repeat(pos_df.values, n_channels, axis=0),
        columns=pos_df.columns
    )

    expanded = pd.DataFrame({
        "TRACE": trace_idx,
        "CHANNEL": chan_idx
    }).join(pos_repeated.reset_index(drop=True))

    return expanded




def expand_positions_with_offsets(pos_df, rad_header):
    """
    Expand pos_df positions for each channel, applying offsets from RAD file.
    
    pos_df: DataFrame with TRACE, NORTHING, EASTING, ELEVATION
    rad_header: dict from your parsed .rad file
    """
    n_channels = int(rad_header.get("CHANNELS", 1))
    x_offsets = np.array(rad_header.get("CH_X_OFFSETS", [0.0]))
    y_offsets = np.array(rad_header.get("CH_Y_OFFSETS", [0.0]))

    if len(x_offsets) != n_channels or len(y_offsets) != n_channels:
        raise ValueError("Mismatch between CHANNELS and CH_X/Y_OFFSETS length")

    # Repeat each trace for each channel
    expanded = []
    for _, row in pos_df.iterrows():
        for ch in range(n_channels):
            expanded.append({
                "TRACE": int(row["TRACE"]),
                "CHANNEL": ch,
                "Y": row["Y"] + y_offsets[ch],  # apply Y offset
                "X": row["X"] + x_offsets[ch],   # apply X offset
                "Z": row["Z"]
            })
    return pd.DataFrame(expanded)


def parse_mala_rad(rad_path: str | Path) -> dict:
    """
    Parse MALÅ .rad header (colon- or equals-separated).
    Returns a dict with UPPERCASE keys. Vector fields become numpy arrays.
    """
    rad_path = Path(rad_path)
    header = {}
    with rad_path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue

            # split on first ":" or "=", whichever comes first
            m = re.match(r"^\s*([^:=]+)\s*[:=]\s*(.*)$", line)
            if not m:
                # line without separator -> store as-is under "__RAW__"
                header.setdefault("__RAW__", []).append(line)
                continue

            key, val = m.group(1).strip().upper(), _clean_value(m.group(2))

            # Some fields are known to be space-separated lists
            if key.startswith("CH_") or key.endswith("_OFFSETS"):
                header[key] = _parse_list(val or "")
            else:
                # Convert to number when reasonable
                if val is None:
                    header[key] = None
                else:
                    # If it looks like a list (contains spaces), try list parse
                    if " " in val and not re.search(r"[A-Za-z]", val):
                        # numbers separated by spaces
                        arr = _parse_list(val)
                        header[key] = arr
                    else:
                        header[key] = _try_number(val)

    # Normalize a few aliases sometimes seen across datasets
    if "NUMBER_OF_CH" in header and "CHANNELS" not in header:
        header["CHANNELS"] = int(header["NUMBER_OF_CH"])
    return header

def _paired_rad_path(data_path: Path) -> Path:
    base = data_path.with_suffix("")
    for ext in (".rad", ".RAD"):
        p = base.with_suffix(ext)
        if p.exists():
            return p
    # sometimes base name differs only by case
    for cand in data_path.parent.glob("*.rad"):
        if cand.stem.lower() == data_path.stem.lower():
            return cand
    raise FileNotFoundError(f"No matching .rad header found for {data_path}")

def _dtype_from_ext(path: Path) -> np.dtype:
    ext = path.suffix.lower()
    if ext == ".rd7":
        return np.dtype("<i4")  # 32-bit little-endian signed int
    if ext == ".rd3":
        return np.dtype("<i2")  # 16-bit little-endian signed int
    # default to RD7-style
    return np.dtype("<i4")

# ---------- Main reader ----------

def read_mala_rd(data_file: str | Path, rad_file: str | Path | None = None,
                 return_shape="t,c,s"):
    """
    Read MALÅ RD3/RD7 data with its RAD header.

    Parameters
    ----------
    data_file : str|Path
        Path to .rd7 or .rd3 file.
    rad_file : str|Path|None
        Path to .rad header. If None, will search next to data_file.
    return_shape : {"t,c,s", "t,s,c", "flat"}
        Desired output axis order:
        - "t,c,s": (n_traces, n_channels, n_samples)  [default]
        - "t,s,c": (n_traces, n_samples, n_channels)
        - "flat":  (n_traces*n_channels, n_samples)

    Returns
    -------
    data : np.ndarray
    header : dict
    """
    data_path = Path(data_file)
    if rad_file is None:
        rad_path = _paired_rad_path(data_path)
    else:
        rad_path = Path(rad_file)

    header = parse_mala_rad(rad_path)

    # Required header fields
    if "SAMPLES" not in header or header["SAMPLES"] is None:
        raise ValueError("Header missing SAMPLES")
    ns = int(header["SAMPLES"])

    n_channels = int(header.get("CHANNELS", 1) or 1)

    dtype = _dtype_from_ext(data_path)
    bps = dtype.itemsize

    # Read raw
    raw = np.fromfile(data_path, dtype=dtype)
    if raw.size == 0:
        raise ValueError(f"No data read from {data_path}")

    # Infer counts from file size
    total_values = raw.size
    if total_values % ns != 0:
        raise ValueError(
            f"File length ({total_values}) is not divisible by SAMPLES ({ns}). "
            "Header might be wrong or file corrupted."
        )

    traces_times_channels = total_values // ns

    if traces_times_channels % n_channels != 0:
        # Try to re-interpret: sometimes CHANNELS is absent or wrong; fall back to 1
        if n_channels != 1 and (traces_times_channels % n_channels) != 0:
            raise ValueError(
                f"Total trace blocks ({traces_times_channels}) not divisible by "
                f"CHANNELS ({n_channels}). Check header 'CHANNELS' or file."
            )

    n_traces = traces_times_channels // n_channels

    # Reshape: blocks are typically stored as [trace0_ch0_samples][trace0_ch1_samples]...[trace1_ch0_samples]...
    data = raw.reshape(n_traces, n_channels, ns)

    # Optional sanity check against LAST TRACE (often last index, so n_traces ≈ LAST TRACE + 1)
    if "LAST TRACE" in header and isinstance(header["LAST TRACE"], int):
        # only warn if it's way off (not fatal)
        expected = header["LAST TRACE"] + 1
        if expected not in (n_traces, n_traces * n_channels):  # depending on vendor meaning
            # Just attach a note; don't raise
            header["_NOTE_LAST_TRACE_MISMATCH"] = {
                "header_last_trace_plus_one": expected,
                "derived_n_traces": n_traces,
                "channels": n_channels
            }

    if return_shape == "t,c,s":
        pass
    elif return_shape == "t,s,c":
        data = np.transpose(data, (0, 2, 1))
    elif return_shape == "flat":
        data = data.reshape(n_traces * n_channels, ns)
    else:
        raise ValueError("return_shape must be 't,c,s', 't,s,c', or 'flat'")

    # Attach a few convenient computed fields
    header["_DERIVED_"] = {
        "dtype": str(dtype),
        "bytes_per_sample": bps,
        "n_samples": ns,
        "n_channels": n_channels,
        "n_traces": n_traces,
        "total_values": int(total_values)
    }
    return data, header
    



def read_mala_data(datafile,radfile, posfile):
    """
    Read MALÅ GPR data and associated position file.

    Parameters
    ----------
    datafile: str or Path
        Path to the .rd3/.rd7 data file.
    radfile : str or Path
        Path to the .rad header file (or .rd3/.rd7 data file; expects .rad next to it).
    posfile : str or Path
        Path to the position file (.cor, .pos, .gps, etc.).

    Returns
    -------
    data : np.ndarray
        The GPR data array.
    hdr : dict
        Parsed header information from the .rad file.
    expanded_df : pd.DataFrame
        DataFrame containing trace positions.

    Expected file types: .rad (header), .rd3/.rd7 (data), .cor/.pos/.gps (positions).
    """
    data, hdr = read_mala_rd(datafile)  # expects yourfile.rad next to it
    pos_df = read_positions(posfile)   # or .pos, .gps, etc.
    expanded_df = expand_positions_with_offsets(pos_df, hdr)
    print(hdr["_DERIVED_"])  # useful derived counts
    return data, hdr, expanded_df
