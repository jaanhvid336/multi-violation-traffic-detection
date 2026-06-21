import pandas as pd
import matplotlib.pyplot as plt

# Shared professional palette (navy / blue / accent set), no emojis.
PALETTE = ["#2563EB", "#F97316", "#10B981", "#8B5CF6", "#EF4444", "#0EA5E9"]
INK = "#1E293B"
GRID = "#E2E8F0"
FACE = "#FFFFFF"


def _style_axes(ax):
    ax.set_facecolor(FACE)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(GRID)
    ax.tick_params(colors=INK, labelsize=9)
    ax.yaxis.label.set_color(INK)
    ax.xaxis.label.set_color(INK)
    ax.title.set_color(INK)


def generate_violation_pie_chart(df):
    if df is None or df.empty:
        return None

    counts = df["Violation Type"].value_counts()

    fig, ax = plt.subplots(figsize=(5.2, 4), dpi=120)
    fig.patch.set_facecolor(FACE)

    wedges, texts, autotexts = ax.pie(
        counts,
        labels=None,
        autopct="%1.1f%%",
        startangle=90,
        colors=PALETTE[:len(counts)],
        pctdistance=0.78,
        wedgeprops=dict(width=0.42, edgecolor=FACE, linewidth=2),
    )
    for t in autotexts:
        t.set_color(INK)
        t.set_fontsize(9)
        t.set_fontweight("bold")

    ax.legend(wedges, counts.index, loc="center left",
              bbox_to_anchor=(0.98, 0.5), frameon=False,
              fontsize=9, labelcolor=INK)
    ax.set_title("Violation Distribution", fontsize=13, fontweight="bold", pad=12)
    ax.axis("equal")
    plt.tight_layout()
    return fig


def generate_type_bar_chart(df):
    if df is None or df.empty:
        return None

    counts = df["Violation Type"].value_counts()

    fig, ax = plt.subplots(figsize=(6, 4), dpi=120)
    fig.patch.set_facecolor(FACE)
    _style_axes(ax)

    bars = ax.bar(range(len(counts)), counts.values,
                  color=PALETTE[:len(counts)], width=0.62, zorder=3)
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels(counts.index, rotation=20, ha="right")
    ax.yaxis.grid(True, color=GRID, zorder=0)
    ax.set_ylabel("Count")
    ax.set_title("Violations by Type", fontsize=13, fontweight="bold", pad=12)
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.1,
                int(b.get_height()), ha="center", va="bottom",
                fontsize=9, color=INK, fontweight="bold")
    plt.tight_layout()
    return fig


def generate_location_bar_chart(df):
    if df is None or df.empty or "Location" not in df.columns:
        return None

    counts = df["Location"].value_counts().head(10).sort_values()

    fig, ax = plt.subplots(figsize=(6, 4), dpi=120)
    fig.patch.set_facecolor(FACE)
    _style_axes(ax)

    ax.barh(range(len(counts)), counts.values, color="#2563EB",
            height=0.62, zorder=3)
    ax.set_yticks(range(len(counts)))
    ax.set_yticklabels(counts.index)
    ax.xaxis.grid(True, color=GRID, zorder=0)
    ax.set_xlabel("Count")
    ax.set_title("Top Locations", fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    return fig


def generate_daily_bar_chart(df):
    if df is None or df.empty:
        return None
    try:
        d = df.copy()
        d["Date"] = pd.to_datetime(d["Timestamp"]).dt.date
        daily = d.groupby("Date").size()

        fig, ax = plt.subplots(figsize=(6, 4), dpi=120)
        fig.patch.set_facecolor(FACE)
        _style_axes(ax)

        ax.plot(range(len(daily)), daily.values, color="#2563EB",
                marker="o", linewidth=2.2, markersize=5, zorder=3)
        ax.fill_between(range(len(daily)), daily.values, color="#2563EB",
                        alpha=0.10, zorder=2)
        ax.set_xticks(range(len(daily)))
        ax.set_xticklabels([str(x) for x in daily.index], rotation=45, ha="right")
        ax.yaxis.grid(True, color=GRID, zorder=0)
        ax.set_ylabel("Violations")
        ax.set_title("Violations Over Time", fontsize=13, fontweight="bold", pad=12)
        plt.tight_layout()
        return fig
    except Exception:
        return None


def generate_timeslot_bar_chart(df):
    if df is None or df.empty or "Time Slot" not in df.columns:
        return None
    counts = df["Time Slot"].value_counts().sort_index()
    if counts.empty:
        return None

    fig, ax = plt.subplots(figsize=(6, 4), dpi=120)
    fig.patch.set_facecolor(FACE)
    _style_axes(ax)

    ax.bar(range(len(counts)), counts.values, color="#8B5CF6",
           width=0.62, zorder=3)
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels(counts.index, rotation=45, ha="right")
    ax.yaxis.grid(True, color=GRID, zorder=0)
    ax.set_ylabel("Count")
    ax.set_title("Violations by Time Slot", fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    return fig
