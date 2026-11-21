"""Plot first-request and cached follow-up latency from the agent session JSON."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    arms = [("release", "GGUF · v0.1"), ("combined-w7", "GGUF · v0.1 + DSpark")]
    if all((args.evidence / f"fused-w7-{size}" / "session.json").exists() for size in ("32k", "64k")):
        arms.append(("fused-w7", "GGUF · fused mask + DSpark"))
    arms.append(("native", "Native SGLang · DSpark"))
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 5.9), sharex=True)
    first_color, cached_color = "#246381", "#40a98b"
    for ax, (size, title) in zip(axes, [("32k", "32,768 input content tokens"), ("64k", "65,536 input content tokens")]):
        for index, (arm, label) in enumerate(arms):
            run = json.loads((args.evidence / f"{arm}-{size}" / "session.json").read_text())
            if run["status"] != "passed" or len(run["turns"]) != 17:
                raise ValueError(f"Incomplete or incorrect session for {arm}-{size}")
            first = run["first_request_wall_s"]
            end = run["session_wall_s"]
            ax.barh(index, first, height=.56, color=first_color)
            ax.barh(index, end - first, left=first, height=.56, color=cached_color)
            ax.text(end + .75, index, f"{end:.1f} s", va="center", fontsize=10)
        ax.set_yticks(range(len(arms)), [label for _, label in arms])
        ax.invert_yaxis()
        ax.set_xlim(0, 82)
        ax.set_axisbelow(True)
        ax.grid(axis="x", linewidth=.5, alpha=.4)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.set_title(title, loc="left", fontsize=11)
    axes[1].set_xlabel("Time to a verified answer from a ready server (seconds)")
    fig.suptitle("Inkling 2 Mini on one RTX 5090", x=.075, ha="left", fontsize=17, fontweight="bold")
    fig.legend(handles=[Patch(color=first_color, label="First request with cold prompt"),
                        Patch(color=cached_color, label="Remaining 16 requests and client overhead")],
               bbox_to_anchor=(.07, .925), loc="center left", ncol=2, frameon=False)
    fig.text(.075, .018, "One sequential tool session per bar. GGUF IQ4_NL / F16 KV. Native mixed NVFP4/MXFP8. DSpark width 7.",
             fontsize=9)
    fig.subplots_adjust(left=.30, right=.96, top=.83, bottom=.13, hspace=.45)
    fig.savefig(args.output, dpi=160)


if __name__ == "__main__":
    main()
