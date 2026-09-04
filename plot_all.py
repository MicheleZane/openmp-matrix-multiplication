"""
Full set of plots for the OpenMP matrix multiplication project.
Generates topic-based figures saved in results/plots/

Y-axis error bars derived from the STDDEV column (standard deviation
of the parallel time over 10 runs). Error propagation:
  sigma_speedup    = speedup    * sigma_T / T_par
  sigma_efficiency = efficiency * sigma_T / T_par
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# -- Setup --------------------------------------------------------------------
RESULT_DIR = "results"
PLOT_DIR   = os.path.join(RESULT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 130,
})

SIZES   = [512, 1024, 2048]
PALETTE = {"512": "#2196F3", "1024": "#F44336", "2048": "#4CAF50"}

PER_SIZE_COLORS = {512:  "#2196F3", 850:  "#1D9E75",
                   1024: "#F44336", 1520: "#FF9800", 2048: "#4CAF50"}
PER_SIZE_L3     = {512:  "~2 MB/matrix",  850:  "~5.5 MB/matrix",
                   1024: "~8 MB/matrix", 1520: "~17.6 MB/matrix", 2048: "~32 MB/matrix"}
SCHED_COLORS = {"static": "#E91E63", "dynamic": "#FF9800", "guided": "#9C27B0"}
BLOCK_COLORS = {32: "#1565C0", 64: "#2E7D32", 128: "#E65100", 256: "#6A1B9A"}

# Common kwargs for errorbar (line style with markers and error bars)
_EB = dict(lw=2, capsize=3, elinewidth=1, capthick=1, markersize=2)

def save(fig, name):
    path = os.path.join(PLOT_DIR, name)
    fig.savefig(path, bbox_inches="tight")
    print(f"  Saved: {path}")

# -- Helper: error propagation for speedup and efficiency ---------------------
# par_col = name of the parallel-time column ("PAR_TIME" or "TIME")
def _yerr_sp(sub, par_col="PAR_TIME"):
    return (sub["SPEEDUP"]    * sub["STDDEV"] / sub[par_col]).values

def _yerr_eff(sub, par_col="PAR_TIME"):
    return (sub["EFFICIENCY"] * sub["STDDEV"] / sub[par_col]).values

# -- Load data ------------------------------------------------------------------
df_base  = pd.read_csv(os.path.join(RESULT_DIR, "results.csv"))
df_scale = pd.read_csv(os.path.join(RESULT_DIR, "thread_scaling.csv"))
df_sched = pd.read_csv(os.path.join(RESULT_DIR, "scheduling_results.csv"))
df_tile_all = pd.read_csv(os.path.join(RESULT_DIR, "tiling_results.csv"))
for _col in ["SIZE", "BLOCK_SIZE", "THREADS", "TIME", "SPEEDUP", "EFFICIENCY", "STDDEV"]:
    if _col in df_tile_all.columns:
        df_tile_all[_col] = pd.to_numeric(df_tile_all[_col], errors="coerce")
df_tile     = df_tile_all[df_tile_all["MODE"] == "tiled"].copy()
df_tile_opt = df_tile_all[df_tile_all["MODE"] == "tiled_opt"].copy()

print("=== Generating plots ===")

# ══════════════════════════════════════════════════════════════════════════
# FIGURE 0 - Thread scaling: from sequential (1) up to 32 threads
# ══════════════════════════════════════════════════════════════════════════
print("\n[0/6] Thread scaling")
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Thread Scaling - Speedup & Efficiency (sequential to 32 threads)",
             fontsize=13, fontweight="bold")

ax1, ax2 = axes
ax1.plot([1, 32], [1, 32], "k--", lw=1, alpha=0.35, label="Ideal")

for size, color in PALETTE.items():
    sub = (df_base[(df_base["SIZE"] == int(size)) &
                   (df_base["THREADS"] >= 2) &
                   (df_base["THREADS"] <= 32)]
           .sort_values("THREADS"))
    t = sub["THREADS"].values
    # sequential point: speedup=1, efficiency=1, yerr=0
    t_plot  = np.concatenate([[1], t])
    sp_plot = np.concatenate([[1.0], sub["SPEEDUP"].values])
    sp_err  = np.concatenate([[0],   _yerr_sp(sub)])
    ef_plot = np.concatenate([[1.0], sub["EFFICIENCY"].values])
    ef_err  = np.concatenate([[0],   _yerr_eff(sub)])

    ax1.errorbar(t_plot, sp_plot, yerr=sp_err,
                 marker="o", color=color, label=f"{size}x{size}", **_EB)
    ax2.errorbar(t_plot, ef_plot, yerr=ef_err,
                 marker="s", color=color, label=f"{size}x{size}", **_EB)

ax1.set_title("Speedup vs Threads")
ax1.set_xlabel("Threads")
ax1.set_ylabel("Speedup")
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.xaxis.set_major_locator(ticker.MultipleLocator(2))

ax2.axhline(1.0, color="k", lw=1, ls="--", alpha=0.35, label="Ideal (100%)")
ax2.set_title("Efficiency vs Threads")
ax2.set_xlabel("Threads")
ax2.set_ylabel("Efficiency")
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.xaxis.set_major_locator(ticker.MultipleLocator(2))
ax2.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))

plt.tight_layout()
save(fig, "0_thread_scaling.png")
plt.close()

# ══════════════════════════════════════════════════════════════════════════
# FIGURE 1 - Base parallel: Speedup & Efficiency (sequential to 32 threads)
# ══════════════════════════════════════════════════════════════════════════
print("[1/6] Base parallel: speedup & efficiency")
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
fig.suptitle("Base Parallel (OpenMP) - Speedup & Efficiency (sequential to 32 threads)",
             fontsize=13, fontweight="bold")

for col, size in enumerate(SIZES):
    sub = (df_base[(df_base["SIZE"] == size) &
                   (df_base["THREADS"] >= 2) &
                   (df_base["THREADS"] <= 32)]
           .sort_values("THREADS"))
    c       = PALETTE[str(size)]
    threads = sub["THREADS"].values

    # sequential point: speedup=1, efficiency=1, yerr=0
    t_plot  = np.concatenate([[1], threads])
    sp_plot = np.concatenate([[1.0], sub["SPEEDUP"].values])
    sp_err  = np.concatenate([[0],   _yerr_sp(sub)])
    ef_plot = np.concatenate([[1.0], sub["EFFICIENCY"].values])
    ef_err  = np.concatenate([[0],   _yerr_eff(sub)])

    # --- Speedup ---
    ax = axes[0, col]
    ax.plot([1, 32], [1, 32], "k--", lw=1, alpha=0.35, label="Ideal")
    ax.errorbar(t_plot, sp_plot, yerr=sp_err,
                marker="o", color=c, label=f"{size}x{size}", **_EB)
    ax.fill_between(t_plot, sp_plot, alpha=0.10, color=c)
    ax.set_title(f"Speedup - {size}x{size}")
    ax.set_xlabel("Threads")
    ax.set_ylabel("Speedup")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(4))

    # --- Efficiency ---
    ax = axes[1, col]
    ax.axhline(1.0, color="k", lw=1, ls="--", alpha=0.35, label="Ideal (100%)")
    ax.errorbar(t_plot, ef_plot, yerr=ef_err,
                marker="s", color=c, **_EB)
    ax.fill_between(t_plot, ef_plot, alpha=0.10, color=c)
    ax.set_title(f"Efficiency - {size}x{size}")
    ax.set_xlabel("Threads")
    ax.set_ylabel("Efficiency")
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(4))

plt.tight_layout()
save(fig, "1_parallel_base.png")
plt.close()

# ══════════════════════════════════════════════════════════════════════════
# FIGURE 2 - Scheduling comparison: static vs dynamic vs guided
# ══════════════════════════════════════════════════════════════════════════
print("[2/6] Scheduling comparison")
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
fig.suptitle("OpenMP Scheduling Comparison - Speedup & Efficiency", fontsize=13, fontweight="bold")

for col, size in enumerate(SIZES):
    sub = df_sched[df_sched["SIZE"] == size]
    for sched, color in SCHED_COLORS.items():
        s = sub[sub["SCHEDULE_TYPE"] == sched].sort_values("THREADS")
        if s.empty:
            continue
        t = s["THREADS"].values

        # Speedup
        axes[0, col].errorbar(t, s["SPEEDUP"], yerr=_yerr_sp(s, "TIME"),
                              marker="o", color=color, label=sched, **_EB)
        # Efficiency
        axes[1, col].errorbar(t, s["EFFICIENCY"], yerr=_yerr_eff(s, "TIME"),
                              marker="s", color=color, label=sched, **_EB)

    # Speedup decorations
    ax = axes[0, col]
    t_range = sorted(sub["THREADS"].unique())
    ax.plot(t_range, t_range, "k--", lw=1, alpha=0.35, label="Ideal")
    ax.set_title(f"Speedup - {size}x{size}")
    ax.set_xlabel("Threads")
    ax.set_ylabel("Speedup")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(4))

    # Efficiency decorations
    ax = axes[1, col]
    ax.axhline(1.0, color="k", lw=1, ls="--", alpha=0.35, label="Ideal")
    ax.set_title(f"Efficiency - {size}x{size}")
    ax.set_xlabel("Threads")
    ax.set_ylabel("Efficiency")
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(4))

plt.tight_layout()
save(fig, "2_scheduling.png")
plt.close()

# ══════════════════════════════════════════════════════════════════════════
# FIGURE 3 - Tiling: Speedup vs Threads for each block size
# ══════════════════════════════════════════════════════════════════════════
print("[3/6] Tiling speedup vs threads")
blocks = sorted(df_tile["BLOCK_SIZE"].unique())
_variants3 = [(df_tile, "Tiled"), (df_tile_opt, "Tiled Opt")]

for size in SIZES:
    fig, axes = plt.subplots(2, len(blocks), figsize=(16, 7), sharey="row")
    fig.suptitle(f"Tiling - Speedup vs Threads  {size}x{size}  (Tiled / Tiled Opt per block size)",
                 fontsize=13, fontweight="bold")
    for var_idx, (df_var, var_label) in enumerate(_variants3):
        for col, block in enumerate(blocks):
            ax  = axes[var_idx, col]
            sub = df_var[(df_var["SIZE"] == size) & (df_var["BLOCK_SIZE"] == block)].sort_values("THREADS")
            c   = BLOCK_COLORS[block]
            if not sub.empty:
                t = sub["THREADS"].values
                ax.plot(t, t, "k--", lw=1, alpha=0.3)
                ax.errorbar(t, sub["SPEEDUP"], yerr=_yerr_sp(sub, "TIME"),
                            marker="o", color=c, **_EB)
                ax.fill_between(t, sub["SPEEDUP"], alpha=0.10, color=c)
            ax.set_title(f"{var_label}  block={block}", fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.xaxis.set_major_locator(ticker.MultipleLocator(8))
            if col == 0:
                ax.set_ylabel("Speedup")
            if var_idx == 1:
                ax.set_xlabel("Threads")
    plt.tight_layout()
    save(fig, f"3_tiling_threads_{size}.png")
    plt.close()

# ══════════════════════════════════════════════════════════════════════════
# FIGURE 4 - Tiling: effect of block size (at fixed thread counts)
# ══════════════════════════════════════════════════════════════════════════
print("[4/6] Tiling: effect of block size")
fixed_threads = [4, 8, 12]
thread_colors = {4: "#1976D2", 8: "#388E3C", 12: "#D32F2F"}
_variants4 = [(df_tile, "Tiled"), (df_tile_opt, "Tiled Opt")]

fig, axes = plt.subplots(2, 3, figsize=(14, 9))
fig.suptitle("Tiling - Speedup vs Block Size (fixed thread counts)", fontsize=13, fontweight="bold")

for row_idx, (df_var, var_label) in enumerate(_variants4):
    for col, size in enumerate(SIZES):
        ax  = axes[row_idx, col]
        sub = df_var[df_var["SIZE"] == size]
        for t_fixed, color in thread_colors.items():
            row = sub[sub["THREADS"] == t_fixed].sort_values("BLOCK_SIZE")
            if row.empty:
                continue
            ax.errorbar(row["BLOCK_SIZE"], row["SPEEDUP"],
                        yerr=_yerr_sp(row, "TIME"),
                        marker="D", color=color, label=f"{t_fixed} threads", **_EB)
        ax.set_title(f"{var_label} - {size}x{size}")
        ax.set_xlabel("Block Size")
        ax.set_ylabel("Speedup")
        ax.set_xticks(blocks)
        ax.legend()
        ax.grid(True, alpha=0.3)

plt.tight_layout()
save(fig, "4_tiling_blocksize.png")
plt.close()

# ══════════════════════════════════════════════════════════════════════════
# FIGURE 5 - Heatmap: Tiling speedup (block size x threads) per size
# ══════════════════════════════════════════════════════════════════════════
print("[5/6] Tiling heatmap")
_variants5 = [(df_tile, "Tiled"), (df_tile_opt, "Tiled Opt")]
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle("Speedup Heatmap - Tiling / Tiled Opt (Block Size x Threads)", fontsize=13, fontweight="bold")

for row_idx, (df_var, var_label) in enumerate(_variants5):
    for col, size in enumerate(SIZES):
        ax  = axes[row_idx, col]
        sub = df_var[df_var["SIZE"] == size]
        if sub.empty:
            ax.set_visible(False)
            continue
        pivot = sub.pivot(index="BLOCK_SIZE", columns="THREADS", values="SPEEDUP")
        im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd",
                       vmin=pivot.values.min(), vmax=pivot.values.max())
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, fontsize=8)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_xlabel("Threads")
        ax.set_ylabel("Block Size")
        ax.set_title(f"{var_label} - {size}x{size}")
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                ax.text(j, i, f"{pivot.values[i, j]:.1f}",
                        ha="center", va="center", fontsize=7,
                        color="black" if pivot.values[i, j] < pivot.values.max() * 0.7 else "white")
        plt.colorbar(im, ax=ax, label="Speedup")

plt.tight_layout()
save(fig, "5_heatmap_tiling.png")
plt.close()

# ══════════════════════════════════════════════════════════════════════════
# FIGURE 6 - Global comparison: best result of each method per size
# ══════════════════════════════════════════════════════════════════════════
print("[6/6] Global comparison")

records = []

for size in SIZES:
    # base
    best = df_base[df_base["SIZE"] == size]["SPEEDUP"].max()
    records.append({"size": size, "method": "Parallel\nBase", "speedup": best})
    # scheduling
    for sched in ["static", "dynamic", "guided"]:
        sub  = df_sched[(df_sched["SIZE"] == size) & (df_sched["SCHEDULE_TYPE"] == sched)]
        best = sub["SPEEDUP"].max()
        records.append({"size": size, "method": sched.capitalize(), "speedup": best})
    # tiling
    sub  = df_tile[df_tile["SIZE"] == size]
    if not sub.empty:
        best_cfg = sub.loc[sub["SPEEDUP"].idxmax()]
        records.append({"size": size, "method": "Tiling",
                        "speedup": best_cfg["SPEEDUP"],
                        "block": int(best_cfg["BLOCK_SIZE"])})
    # tiled_opt (only if present in the CSV)
    sub_opt = df_tile_opt[df_tile_opt["SIZE"] == size]
    if not sub_opt.empty:
        best_cfg = sub_opt.loc[sub_opt["SPEEDUP"].idxmax()]
        records.append({"size": size, "method": "Tiling Opt",
                        "speedup": best_cfg["SPEEDUP"],
                        "block": int(best_cfg["BLOCK_SIZE"])})

df_cmp = pd.DataFrame(records)

methods  = df_cmp["method"].unique()
x        = np.arange(len(methods))
width    = 0.25
bar_cols = list(PALETTE.values())

fig, ax = plt.subplots(figsize=(13, 6))
fig.suptitle("Global Comparison - Best Speedup per Method and Size",
             fontsize=13, fontweight="bold")

for i, (size, color) in enumerate(zip(SIZES, bar_cols)):
    vals = [df_cmp[(df_cmp["size"] == size) & (df_cmp["method"] == m)]["speedup"].values
            for m in methods]
    vals = [v[0] if len(v) > 0 else 0 for v in vals]
    bars = ax.bar(x + (i - 1) * width, vals, width, label=f"{size}x{size}", color=color, alpha=0.85)
    for bar, val, m in zip(bars, vals, methods):
        if val == 0:
            continue
        row_match = df_cmp[(df_cmp["size"] == size) & (df_cmp["method"] == m)]
        block = row_match["block"].values
        label = f"{val:.1f}x"
        if len(block) > 0 and not pd.isna(block[0]):
            label += f"\nb={int(block[0])}"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                label, ha="center", va="bottom", fontsize=7)

ax.set_xticks(x)
ax.set_xticklabels(methods, fontsize=9)
ax.set_ylabel("Best Speedup")
ax.set_xlabel("Method")
ax.legend(title="Size")
ax.grid(True, axis="y", alpha=0.3)
ax.set_ylim(0, df_cmp["speedup"].max() * 1.15)

plt.tight_layout()
save(fig, "6_global_comparison.png")
plt.close()

# ══════════════════════════════════════════════════════════════════════════
# FIGURE 7 - Per-size plots: Time, Speedup, Efficiency
#             (sequential to 32 threads)
# ══════════════════════════════════════════════════════════════════════════
print("[7/7] Per-size plots")

sizes_present = sorted(df_base["SIZE"].unique())

for size in sizes_present:
    sub = (df_base[(df_base["SIZE"] == size) &
                   (df_base["THREADS"] >= 2) &
                   (df_base["THREADS"] <= 32)]
           .sort_values("THREADS"))
    threads  = sub["THREADS"].values
    seq_time = df_base[df_base["SIZE"] == size]["SEQ_TIME"].iloc[0]
    color    = PER_SIZE_COLORS.get(size, "#607D8B")
    l3_note  = PER_SIZE_L3.get(size, "")

    # array with the sequential point at x=1
    t_plot    = np.concatenate([[1], threads])
    time_plot = np.concatenate([[seq_time], sub["PAR_TIME"].values])
    time_err  = np.concatenate([[0],        sub["STDDEV"].values])
    sp_plot   = np.concatenate([[1.0],      sub["SPEEDUP"].values])
    sp_err    = np.concatenate([[0],        _yerr_sp(sub)])
    ef_plot   = np.concatenate([[1.0],      sub["EFFICIENCY"].values])
    ef_err    = np.concatenate([[0],        _yerr_eff(sub)])

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle(
        f"Size {size}x{size}   ({l3_note}) - sequential to 32 threads",
        fontsize=13, fontweight="bold"
    )

    # Parallel time
    ax = axes[0]
    ax.errorbar(t_plot, time_plot, yerr=time_err,
                marker="o", color=color, label="Time", **_EB)
    ax.fill_between(t_plot, time_plot, alpha=0.12, color=color)
    ax.axvline(6, color="gray", lw=1, ls="--", alpha=0.5, label="6 physical cores")
    ax.set_title("Execution Time")
    ax.set_xlabel("Threads  (1 = sequential)")
    ax.set_ylabel("Time (s)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(4))

    # Speedup
    ax = axes[1]
    ax.plot([1, 32], [1, 32], "k--", lw=1, alpha=0.35, label="Ideal")
    ax.errorbar(t_plot, sp_plot, yerr=sp_err,
                marker="o", color=color, label="Actual", **_EB)
    ax.fill_between(t_plot, sp_plot, alpha=0.12, color=color)
    ax.axvline(6, color="gray", lw=1, ls="--", alpha=0.5, label="6 physical cores")
    ax.set_title("Speedup")
    ax.set_xlabel("Threads  (1 = sequential)")
    ax.set_ylabel("Speedup")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(4))

    # Efficiency
    ax = axes[2]
    ax.axhline(1.0, color="k", lw=1, ls="--", alpha=0.35, label="Ideal (100%)")
    ax.errorbar(t_plot, ef_plot, yerr=ef_err,
                marker="s", color=color, label="Actual", **_EB)
    ax.fill_between(t_plot, ef_plot, alpha=0.12, color=color)
    ax.axvline(6, color="gray", lw=1, ls="--", alpha=0.5, label="6 physical cores")
    ax.set_title("Efficiency")
    ax.set_xlabel("Threads  (1 = sequential)")
    ax.set_ylabel("Efficiency")
    ax.set_ylim(0, 1.15)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(4))

    plt.tight_layout()
    save(fig, f"7_per_size_{size}.png")
    plt.close()

# ══════════════════════════════════════════════════════════════════════════
# Shared definitions for Figures 8 and 9
# ══════════════════════════════════════════════════════════════════════════
ALL_SIZES_PALETTE = {
    512:  "#2196F3",
    850:  "#1D9E75",
    1024: "#F44336",
    1520: "#FF9800",
    2048: "#4CAF50",
}

all_sizes = sorted(df_base["SIZE"].unique())

# ══════════════════════════════════════════════════════════════════════════
# FIGURE 8a - results.csv: Parallel time - first 3 sizes (512, 850, 1024)
# ══════════════════════════════════════════════════════════════════════════
print("[8a/9] Parallel time per size (512, 850, 1024)")

sizes_8a = all_sizes[:3]
fig, axes = plt.subplots(len(sizes_8a), 1, figsize=(10, 4 * len(sizes_8a)))
fig.suptitle("Parallel Execution Time vs Threads - sizes 512, 850, 1024",
             fontsize=13, fontweight="bold")

for row, size in enumerate(sizes_8a):
    sub      = df_base[df_base["SIZE"] == size].sort_values("THREADS")
    color    = ALL_SIZES_PALETTE.get(size, "#607D8B")
    t        = sub["THREADS"].values
    seq_time = sub["SEQ_TIME"].iloc[0]

    ax = axes[row]
    ax.errorbar(t, sub["PAR_TIME"], yerr=sub["STDDEV"].values,
                marker="o", color=color, label=f"Parallel {size}x{size}", **_EB)
    ax.fill_between(t, sub["PAR_TIME"], alpha=0.12, color=color)
    ax.axhline(seq_time, color="crimson", lw=2, ls="--", alpha=0.85,
               label=f"Sequential: {seq_time:.2f} s")
    ax.text(t[-1], seq_time, f"  seq={seq_time:.2f}s",
            va="bottom", ha="left", fontsize=8, color="crimson")
    ax.set_title(f"Parallel time - {size}x{size}")
    ax.set_xlabel("Threads")
    ax.set_ylabel("Time (s)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(8))

plt.tight_layout()
save(fig, "8a_par_time_per_size_small.png")
plt.close()

# ══════════════════════════════════════════════════════════════════════════
# FIGURE 8b - results.csv: Parallel time - last 2 sizes (1520, 2048)
# ══════════════════════════════════════════════════════════════════════════
print("[8b/9] Parallel time per size (1520, 2048)")

sizes_8b = all_sizes[3:]
fig, axes = plt.subplots(len(sizes_8b), 1, figsize=(10, 4 * len(sizes_8b)))
fig.suptitle("Parallel Execution Time vs Threads - sizes 1520, 2048",
             fontsize=13, fontweight="bold")

for row, size in enumerate(sizes_8b):
    sub      = df_base[df_base["SIZE"] == size].sort_values("THREADS")
    color    = ALL_SIZES_PALETTE.get(size, "#607D8B")
    t        = sub["THREADS"].values
    seq_time = sub["SEQ_TIME"].iloc[0]

    ax = axes[row]
    ax.errorbar(t, sub["PAR_TIME"], yerr=sub["STDDEV"].values,
                marker="o", color=color, label=f"Parallel {size}x{size}", **_EB)
    ax.fill_between(t, sub["PAR_TIME"], alpha=0.12, color=color)
    ax.axhline(seq_time, color="crimson", lw=2, ls="--", alpha=0.85,
               label=f"Sequential: {seq_time:.2f} s")
    ax.text(t[-1], seq_time, f"  seq={seq_time:.2f}s",
            va="bottom", ha="left", fontsize=8, color="crimson")
    ax.set_title(f"Parallel time - {size}x{size}")
    ax.set_xlabel("Threads")
    ax.set_ylabel("Time (s)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(8))

plt.tight_layout()
save(fig, "8b_par_time_per_size_large.png")
plt.close()

# ══════════════════════════════════════════════════════════════════════════
print(f"\n=== Done! All plots are in: {PLOT_DIR}/ ===")
