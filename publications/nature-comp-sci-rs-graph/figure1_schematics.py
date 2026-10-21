#!/usr/bin/env python3

"""Figure 1, panels B (network schematic) and C (mining workflow)."""

from __future__ import annotations

from pathlib import Path

import evaplot
import matplotlib.pyplot as plt
import utils as u
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.text import Annotation

###############################################################################

INK = "#1e1e1e"
# Identity links are made by ADAMM and article-repository pairs by ARMM; swap these constants
# to recolour both panels.
IDENTITY_COLOR = u.IDENTITY_LINK_COLOR
IDENTITY_TEXT_COLOR = "#137a5c"
PAIR_COLOR = u.SEED_LINK_COLOR
MINED_COLOR = u.MINED_LINK_COLOR
MINED_TEXT_COLOR = u.MINED_LINK_TEXT_COLOR
ADAMM_COLOR = IDENTITY_COLOR
ARMM_COLOR = MINED_COLOR

# ---- Panel B: node and edge data, in schematic units (x right, y up) ----
B_ARTICLES = {"A1": (2.6, 6.8), "A2": (6.0, 8.8), "A3": (7.2, 6.6)}
B_RESEARCHERS = {
    "R1": (0.6, 7.8),
    "R2": (2.6, 8.8),
    "R3": (0.0, 6.0),
    "R4": (5.0, 10.6),
    "R5": (6.8, 10.6),
    "R6": (4.8, 7.2),
    "R7": (8.4, 8.6),
    "R8": (9.2, 6.0),
}
B_REPOSITORIES = {"P1": (2.6, 1.2), "P2": (6.0, 3.2), "P3": (7.2, 1.0)}
B_DEVELOPERS = {"D1": (0.0, 0.4), "D2": (4.8, 1.6), "D3": (9.2, 0.4)}
B_AUTHORSHIP = [
    ("A1", "R1"),
    ("A1", "R2"),
    ("A1", "R3"),
    ("A1", "R6"),
    ("A2", "R4"),
    ("A2", "R5"),
    ("A2", "R6"),
    ("A2", "R7"),
    ("A3", "R6"),
    ("A3", "R7"),
    ("A3", "R8"),
]
B_CONTRIBUTION = [("P1", "D1"), ("P1", "D2"), ("P2", "D2"), ("P3", "D2"), ("P3", "D3")]
B_SEED_PAIR_LINKS = [("A1", "P1"), ("A2", "P2")]
B_MINED_PAIR_LINKS = [("A3", "P3")]
B_IDENTITY_LINKS = [("R3", "D1"), ("R6", "D2"), ("R8", "D3")]
B_DIVIDER_Y = 4.6
B_FIGSIZE = (2.6, 2.63)
# Print size over drawn size; stroke and font targets below are given at print size.
B_PRINT_SCALE = 0.952
B_FONT_SIZE = 6.0 / B_PRINT_SCALE
B_HOLLOW_LW = 0.8 / B_PRINT_SCALE
B_WITHIN_LW = 0.5 / B_PRINT_SCALE
B_CROSS_LW = 0.7 / B_PRINT_SCALE
B_LEGEND_FRAME_LW = 0.4 / B_PRINT_SCALE
B_LAYER_LABEL_COLOR = "#777777"
AUTHORSHIP_COLOR = "#b5b5b5"
CONTRIBUTION_COLOR = "#707070"
# Marker area (pt^2) per node type, tuned so the four glyphs read at similar visual weight.
B_MARKER_AREA = {"article": 62.0, "repository": 52.0, "researcher": 40.0, "developer": 64.0}

# ---- Panel C: sizes in inches ----
# 372 pt text width less the saved padding, so C prints at about 1.0x.
C_WIDTH = 5.127
C_PRINT_SCALE = 1.0
C_FS = 6.0 / C_PRINT_SCALE
C_TITLE_FS = 6.5 / C_PRINT_SCALE
C_LINE_H = 0.098
C_CHIP_H = 0.145
C_CHIP_GAP = 0.06
C_DOTS_H = 0.075
C_BOX_PAD_X = 0.04
C_CHIP_INSET = 0.03
C_CHIP_PAD_X = 0.035
C_GLYPH_STEP = 0.062
C_GLYPH_GAP = 0.026
C_FRAME_LW = 0.7
C_CHIP_LW = 0.45
C_FLOW_LW = 0.5
C_LOOP_LW = 0.6
C_HEAD = "-|>,head_length=0.6,head_width=0.3"
C_MIXED_EDGE = "#8c8c8c"
C_GLYPH_SIZE = {"article": 3.6, "repository": 3.2, "researcher": 3.6, "developer": 3.8}

###############################################################################
# Small drawing helpers

Chip = tuple[str, tuple[str, ...]]


def _arrow(
    ax: Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str = INK,
    both: bool = False,
    lw: float = C_FLOW_LW,
) -> None:
    """Straight arrow from `start` to `end`, with a head at each end if `both`."""
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle=C_HEAD.replace("-|>", "<|-|>") if both else C_HEAD,
            mutation_scale=3.7,
            color=color,
            lw=lw,
            shrinkA=0,
            shrinkB=0,
        )
    )


def _line(
    ax: Axes, points: list[tuple[float, float]], color: str = INK, lw: float = C_FLOW_LW
) -> None:
    """Right-angle polyline through `points`, without an arrowhead."""
    xs, ys = zip(*points, strict=True)
    ax.plot(
        xs, ys, color=color, lw=lw, solid_capstyle="butt", solid_joinstyle="miter", zorder=2
    )


def _elbow(
    ax: Axes, points: list[tuple[float, float]], color: str = INK, lw: float = C_FLOW_LW
) -> None:
    """Right-angle polyline through `points`; the arrowhead sits on the last segment."""
    _line(ax, points[:-1], color=color, lw=lw)
    _arrow(ax, points[-2], points[-1], color=color, lw=lw)


def _use_type42_fonts() -> None:
    """Embed TrueType (Type 42) fonts in PDF and PostScript output."""
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})


def _text_width(ax: Axes, text: str, fontsize: float) -> float:
    """Return the rendered width of `text` in inches."""
    artist = ax.text(0, 0, text, fontsize=fontsize)
    width = artist.get_window_extent().width / ax.figure.dpi
    artist.remove()
    return float(width)


def _chip_width(ax: Axes, chip: Chip) -> float:
    """Chip width (inches): glyphs, label and padding."""
    label, types = chip
    return (
        2 * C_CHIP_PAD_X
        + len(types) * C_GLYPH_STEP
        + C_GLYPH_GAP
        + _text_width(ax, label, C_FS)
    )


def _box_h(head_lines: int, n_chips: int, dots: bool = True) -> float:
    """Height (inches) of a titled box holding `n_chips` chips, plus an ellipsis row."""
    head_h = head_lines * C_LINE_H + 0.07
    return head_h + n_chips * (C_CHIP_H + C_CHIP_GAP) + (C_DOTS_H if dots else 0) + 0.02


def _box_width(ax: Axes, title: str, chips: list[Chip]) -> float:
    """Smallest box width (inches) that holds the title lines and the widest chip."""
    title_w = max(_text_width(ax, line, C_TITLE_FS) for line in title.split("\n"))
    chip_w = max(_chip_width(ax, chip) for chip in chips)
    return max(title_w + 2 * C_BOX_PAD_X, chip_w + 2 * C_CHIP_INSET)


def _glyph(ax: Axes, x: float, y: float, ntype: str) -> None:
    """One node glyph centred on (x, y), styled like panel B's markers."""
    color = u.NODE_BORDER_COLORS[ntype]
    hollow = ntype in u.HOLLOW_NODE_TYPES
    ax.plot(
        [x],
        [y],
        marker=u.NODE_TYPE_MARKERS[ntype],
        ls="None",
        markersize=C_GLYPH_SIZE[ntype],
        markerfacecolor="white" if hollow else color,
        markeredgecolor=color,
        markeredgewidth=0.6 if hollow else 0,
        zorder=5,
    )


def _chip(ax: Axes, x: float, y: float, w: float, chip: Chip) -> None:
    """One chip centred vertically on y: node glyphs at the left, then the label."""
    label, types = chip
    edge = u.NODE_BORDER_COLORS[types[0]] if len(types) == 1 else C_MIXED_EDGE
    ax.add_patch(
        Rectangle(
            (x, y - C_CHIP_H / 2), w, C_CHIP_H, fc="white", ec=edge, lw=C_CHIP_LW, zorder=3
        )
    )
    for i, ntype in enumerate(types):
        _glyph(ax, x + C_CHIP_PAD_X + C_GLYPH_STEP * (i + 0.5), y, ntype)
    ax.text(
        x + C_CHIP_PAD_X + C_GLYPH_STEP * len(types) + C_GLYPH_GAP,
        y,
        label,
        ha="left",
        va="center",
        fontsize=C_FS,
        color=INK,
        zorder=4,
    )


def _title(ax: Axes, x: float, y_top: float, w: float, head_lines: int, title: str) -> float:
    """Centred box title; returns the header height."""
    head_h = head_lines * C_LINE_H + 0.07
    ax.text(
        x + w / 2,
        y_top - head_h / 2,
        title,
        ha="center",
        va="center",
        fontsize=C_TITLE_FS,
        linespacing=1.05,
        color=INK,
        zorder=4,
    )
    return head_h


def _box(
    ax: Axes,
    x: float,
    y_top: float,
    w: float,
    title: str,
    chips: list[Chip],
    head_lines: int,
) -> dict[str, float]:
    """Titled box with two chips and an ellipsis row; returns its edges and centre."""
    h = _box_h(head_lines, len(chips))
    ax.add_patch(Rectangle((x, y_top - h), w, h, fc="white", ec=INK, lw=C_FRAME_LW, zorder=2))
    head_h = _title(ax, x, y_top, w, head_lines, title)
    y_chip0 = y_top - head_h - C_CHIP_GAP / 2 - C_CHIP_H / 2
    for i, chip in enumerate(chips):
        y = y_chip0 - i * (C_CHIP_H + C_CHIP_GAP)
        _chip(ax, x + C_CHIP_INSET, y, w - 2 * C_CHIP_INSET, chip)
    y_dots = y_top - h + 0.02 + C_DOTS_H / 2
    ax.plot(
        [x + w / 2 + d for d in (-0.04, 0, 0.04)],
        [y_dots] * 3,
        marker="o",
        ls="None",
        markersize=1.1,
        color="#777777",
        markeredgewidth=0,
        zorder=4,
    )
    return {"x0": x, "x1": x + w, "y0": y_top - h, "y1": y_top, "yc": y_chip0}


def _pair_box(
    ax: Axes, x: float, y_top: float, w: float, title: str, chips: list[Chip], lines: int
) -> dict[str, float]:
    """One pair or identity: two chips stacked and joined by a short line, laid out on the same
    chip rows as `_box`; returns its edges plus the y of each chip.
    """
    head_h = lines * C_LINE_H + 0.07
    h = head_h + 2 * C_CHIP_H + 2 * C_CHIP_GAP
    ax.add_patch(Rectangle((x, y_top - h), w, h, fc="white", ec=INK, lw=C_FRAME_LW, zorder=2))
    _title(ax, x, y_top, w, lines, title)
    y_a = y_top - head_h - C_CHIP_GAP / 2 - C_CHIP_H / 2
    y_b = y_a - C_CHIP_H - C_CHIP_GAP
    for y, chip in ((y_a, chips[0]), (y_b, chips[1])):
        _chip(ax, x + C_CHIP_INSET, y, w - 2 * C_CHIP_INSET, chip)
    ax.plot([x + w / 2] * 2, [y_a - C_CHIP_H / 2, y_b + C_CHIP_H / 2], color=INK, lw=C_FLOW_LW)
    return {"x0": x, "x1": x + w, "y0": y_top - h, "y1": y_top, "ya": y_a, "yb": y_b}


###############################################################################
# Panel B


def _draw_network_nodes(ax: Axes) -> None:
    """Draw the four node types with panel A's glyphs and colours."""
    groups = {
        "article": B_ARTICLES,
        "researcher": B_RESEARCHERS,
        "repository": B_REPOSITORIES,
        "developer": B_DEVELOPERS,
    }
    for ntype, nodes in groups.items():
        color = u.NODE_TYPE_COLORS[ntype]
        hollow = ntype in u.HOLLOW_NODE_TYPES
        xs, ys = zip(*nodes.values(), strict=True)
        ax.scatter(
            xs,
            ys,
            marker=u.NODE_TYPE_MARKERS[ntype],
            s=B_MARKER_AREA[ntype],
            facecolors="white" if hollow else color,
            edgecolors=color,
            linewidths=B_HOLLOW_LW if hollow else 0,
            zorder=4,
        )


def figure_1b_network_schematic(output_dir: Path = u.OUTPUT_DIR) -> None:
    """
    Build Figure 1, panel B: a two-layer toy network showing the four node types and four
    edge types. The top layer is the article-authorship network, the bottom layer the
    repository-contributor network, and dashed links cross between them.
    """
    evaplot.set_style("evaplot_rc")
    _use_type42_fonts()
    positions = {**B_ARTICLES, **B_RESEARCHERS, **B_REPOSITORIES, **B_DEVELOPERS}
    legend_h = 0.56
    fig = plt.figure(figsize=B_FIGSIZE)
    ax = fig.add_axes((0, legend_h / B_FIGSIZE[1], 1, 1 - legend_h / B_FIGSIZE[1]))
    ax.set_xlim(-0.5, 9.7)
    ax.set_ylim(-0.4, 11.9)
    ax.set_axis_off()

    dashed = (0, (3, 2))
    for edges, color, ls, lw in (
        (B_AUTHORSHIP, AUTHORSHIP_COLOR, "-", B_WITHIN_LW),
        (B_CONTRIBUTION, CONTRIBUTION_COLOR, "-", B_WITHIN_LW),
        (B_SEED_PAIR_LINKS, PAIR_COLOR, dashed, B_CROSS_LW),
        (B_MINED_PAIR_LINKS, MINED_COLOR, dashed, B_CROSS_LW),
        (B_IDENTITY_LINKS, IDENTITY_COLOR, dashed, B_CROSS_LW),
    ):
        for a, b in edges:
            (x0, y0), (x1, y1) = positions[a], positions[b]
            ax.plot([x0, x1], [y0, y1], color=color, ls=ls, lw=lw, zorder=1)
    _draw_network_nodes(ax)

    ax.plot(
        [-0.3, 9.5],
        [B_DIVIDER_Y] * 2,
        color="#777777",
        lw=B_WITHIN_LW,
        ls=(0, (1, 2)),
        zorder=0,
    )
    # Each layer label sits at the upper-left corner of its layer.
    for y, label in (
        (11.7, "Authorship\nnetwork"),
        (B_DIVIDER_Y - 0.25, "Contribution\nnetwork"),
    ):
        ax.text(
            0.3,
            y,
            label,
            ha="left",
            va="top",
            fontsize=B_FONT_SIZE,
            linespacing=1.1,
            color=B_LAYER_LABEL_COLOR,
            zorder=5,
        )

    ax.text(
        7.4,
        3.4,
        "mined",
        ha="left",
        va="center",
        fontsize=B_FONT_SIZE,
        color=MINED_TEXT_COLOR,
        zorder=5,
    )

    def node_handle(ntype: str, label: str) -> Line2D:
        hollow = ntype in u.HOLLOW_NODE_TYPES
        color = u.NODE_TYPE_COLORS[ntype]
        return Line2D(
            [],
            [],
            marker=u.NODE_TYPE_MARKERS[ntype],
            ls="None",
            markersize=B_MARKER_AREA[ntype] ** 0.5,
            markerfacecolor="white" if hollow else color,
            markeredgecolor=color,
            markeredgewidth=B_HOLLOW_LW if hollow else 0,
            label=label,
        )

    def edge_handle(color: str, ls: str | tuple[float, tuple[float, float]] = "-") -> Line2D:
        return Line2D([], [], color=color, lw=B_CROSS_LW + 0.4, ls=ls)

    handles = [
        node_handle("article", "Article"),
        node_handle("repository", "Repository"),
        node_handle("researcher", "Researcher (author)"),
        node_handle("developer", "Developer account"),
        edge_handle(AUTHORSHIP_COLOR),
        edge_handle(CONTRIBUTION_COLOR),
        (edge_handle(PAIR_COLOR, (0, (2.2, 1.4))), edge_handle(MINED_COLOR, (0, (2.2, 1.4)))),
        edge_handle(IDENTITY_COLOR, dashed),
    ]
    pair_label = "Article\u2013repository pair (seed / mined)"
    labels = [
        "Article",
        "Repository",
        "Researcher (author)",
        "Developer account",
        "Authorship",
        "Code contribution",
        pair_label,
        "Researcher\u2013developer identity",
    ]
    legend = fig.legend(
        handles=handles,
        labels=labels,
        handler_map={tuple: HandlerTuple(ndivide=2, pad=0)},
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=2,
        fontsize=B_FONT_SIZE,
        handlelength=1.8,
        columnspacing=1.0,
        markerscale=1.0,
        labelspacing=0.35,
        borderpad=0.4,
    )
    u.style_legend(legend, fontsize=B_FONT_SIZE)
    legend.get_frame().set_edgecolor("#bbbbbb")
    legend.get_frame().set_linewidth(B_LEGEND_FRAME_LW)

    # Legend text is single-colour: hide the pair label and redraw it in coloured chunks.
    pair_text = legend.get_texts()[labels.index(pair_label)]
    base_color = pair_text.get_color()
    pair_text.set_alpha(0)
    anchor: Artist = pair_text
    chunks = [
        ("Article\u2013repository pair (", base_color),
        ("seed", PAIR_COLOR),
        (" / ", base_color),
        ("mined", MINED_TEXT_COLOR),
        (")", base_color),
    ]
    for chunk, color in chunks:
        anchor = fig.add_artist(
            Annotation(
                chunk,
                xy=(0 if anchor is pair_text else 1, 0),
                xycoords=anchor,
                ha="left",
                va="bottom",
                color=color,
                fontsize=B_FONT_SIZE,
                zorder=legend.get_zorder() + 1,
            )
        )

    u.print_caption_note(
        "figure1b_network_schematic",
        "Schematic of the four node types and four edge types.",
    )
    u.save_figure(fig, "figure1b_network_schematic", output_dir, pad_inches=0.02)
    plt.close(fig)


###############################################################################
# Panel C


def _model_label(ax: Axes, x: float, y: float, label: str, color: str) -> None:
    """Right-aligned model name beside its arrow."""
    ax.text(x, y, label, color=color, fontsize=C_FS, fontweight="bold", ha="right", va="center")


def figure_1c_mining_workflow(output_dir: Path = u.OUTPUT_DIR) -> None:
    """
    Build Figure 1, panel C: the iterative mining workflow, left to right in two lanes. The
    article/researcher side runs along the top, the repository/developer side along the
    bottom, and the two matching models (ADAMM, ARMM) sit between the lanes. The loop-back
    arrow leaves ARMM and runs under the whole figure.
    """
    evaplot.set_style("evaplot_rc")
    _use_type42_fonts()
    lane_lines = 2
    lane_h = _box_h(lane_lines, 2)
    lane_gap = 0.38
    loop_gap = 0.17
    loop_h = loop_gap + 0.02
    height = 2 * lane_h + lane_gap + loop_h + 0.06

    fig, ax = plt.subplots(figsize=(C_WIDTH, height))
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set_xlim(0, C_WIDTH)
    ax.set_ylim(0, height)
    ax.set_axis_off()

    top_y = height - 0.03
    bot_y = top_y - lane_h - lane_gap
    mid_y = top_y - lane_h - lane_gap / 2
    loop_y = bot_y - lane_h - loop_gap

    pair_a: Chip = ("article\u2013repository", ("article", "repository"))
    ident: Chip = ("researcher\u2013developer", ("researcher", "developer"))
    researcher: Chip = ("researcher", ("researcher",))
    developer_account: Chip = ("developer account", ("developer",))
    article: Chip = ("article", ("article",))
    repository: Chip = ("repository", ("repository",))
    specs: dict[str, tuple[str, list[Chip]]] = {
        "d1": ("Dataset of\narticle\u2013repository\npairs", [pair_a] * 2),
        "p1": ("Single pair", [article, repository]),
        "top1": ("Article\nresearchers", [researcher] * 2),
        "bot1": ("Repository\ndeveloper accounts", [developer_account] * 2),
        "d2": ("Dataset of\nresearcher\u2013developer\nidentities", [ident] * 2),
        "p2": ("Single identity", [researcher, developer_account]),
        "top2": ("Researcher\narticles", [article] * 2),
        "bot2": ("Developer\nrepositories", [repository] * 2),
    }
    w = {k: _box_width(ax, *spec) for k, spec in specs.items()}
    col_w = [
        w["d1"],
        w["p1"],
        max(w["top1"], w["bot1"]),
        w["d2"],
        w["p2"],
        max(w["top2"], w["bot2"]),
    ]
    left_margin, right_margin = 0.01, 0.075
    gap = (C_WIDTH - left_margin - right_margin - sum(col_w)) / 5
    print(f"Panel C column gap: {gap:.3f} in", [round(c, 3) for c in col_w])
    xs = [left_margin]
    for width in col_w[:-1]:
        xs.append(xs[-1] + width + gap)

    # Dataset and pair boxes share chip rows, centred on mid_y between the two chips.
    row_offset = C_CHIP_GAP + C_CHIP_H
    d_top = mid_y + 3 * C_LINE_H + 0.07 + row_offset
    p_top = mid_y + 1 * C_LINE_H + 0.07 + row_offset
    d1 = _box(ax, xs[0], d_top, col_w[0], *specs["d1"], head_lines=3)
    p1 = _pair_box(ax, xs[1], p_top, col_w[1], *specs["p1"], lines=1)
    top1 = _box(ax, xs[2], top_y, col_w[2], *specs["top1"], head_lines=lane_lines)
    bot1 = _box(ax, xs[2], bot_y, col_w[2], *specs["bot1"], head_lines=lane_lines)
    d2 = _box(ax, xs[3], d_top, col_w[3], *specs["d2"], head_lines=3)
    p2 = _pair_box(ax, xs[4], p_top, col_w[4], *specs["p2"], lines=1)
    top2 = _box(ax, xs[5], top_y, col_w[5], *specs["top2"], head_lines=lane_lines)
    bot2 = _box(ax, xs[5], bot_y, col_w[5], *specs["bot2"], head_lines=lane_lines)

    # Seed input above dataset 1, then dataset 1 -> pair and pair -> the two lanes.
    x_seed = (d1["x0"] + d1["x1"]) / 2
    ax.text(
        x_seed,
        d1["y1"] + 0.2,
        "Seed pairs (5 sources)",
        color=PAIR_COLOR,
        fontsize=C_FS,
        ha="center",
        va="center",
    )
    _arrow(ax, (x_seed, d1["y1"] + 0.12), (x_seed, d1["y1"]), color=PAIR_COLOR)
    _arrow(ax, (d1["x1"], mid_y), (p1["x0"], mid_y))
    # The turn sits near each pair box so the last segment holds the arrowhead clear of it.
    for src, dst, x_turn in (
        (p1, (top1, bot1), p1["x1"] + 0.4 * gap),
        (p2, (top2, bot2), p2["x1"] + 0.4 * gap),
    ):
        for y_src, lane in ((src["ya"], dst[0]), (src["yb"], dst[1])):
            _elbow(
                ax,
                [
                    (src["x1"], y_src),
                    (x_turn, y_src),
                    (x_turn, lane["yc"]),
                    (lane["x0"], lane["yc"]),
                ],
            )

    _arrow(ax, (d2["x1"], mid_y), (p2["x0"], mid_y))

    # Matching models: one double-headed arrow between the lanes, name beside it.
    for top, bot, color, text_color, label in (
        (top1, bot1, ADAMM_COLOR, IDENTITY_TEXT_COLOR, "ADAMM"),
        (top2, bot2, ARMM_COLOR, MINED_TEXT_COLOR, "ARMM"),
    ):
        x_arrow = top["x0"] + 0.62 * (top["x1"] - top["x0"])
        _arrow(
            ax,
            (x_arrow, top["y0"] - 0.03),
            (x_arrow, bot["y1"] + 0.03),
            color=color,
            both=True,
        )
        _model_label(ax, x_arrow - 0.06, mid_y, label, text_color)

    # Outputs leave from the middle of each model arrow: ADAMM into the identity dataset,
    # ARMM down the right edge and left under everything into dataset 1.
    x_adamm = top1["x0"] + 0.62 * (top1["x1"] - top1["x0"])
    _arrow(ax, (x_adamm, mid_y), (d2["x0"], mid_y), color=ADAMM_COLOR, lw=C_LOOP_LW)
    x_arm = top2["x0"] + 0.62 * (top2["x1"] - top2["x0"])
    x_loop = top2["x1"] + 0.06
    x_in = x_seed
    _line(
        ax,
        [(x_arm, mid_y), (x_loop, mid_y), (x_loop, loop_y), (x_in, loop_y)],
        color=ARMM_COLOR,
        lw=C_LOOP_LW,
    )
    _arrow(ax, (x_in, loop_y), (x_in, d1["y0"]), color=ARMM_COLOR, lw=C_LOOP_LW)
    ax.text(
        (x_in + x_loop) / 2,
        loop_y + 0.03,
        "Mined article\u2013repository pairs",
        color=MINED_TEXT_COLOR,
        fontsize=C_FS,
        ha="center",
        va="bottom",
    )

    u.print_caption_note(
        "figure1c_mining_workflow",
        "Iterative mining workflow; ADAMM and ARMM are expanded in the caption.",
    )
    u.save_figure(fig, "figure1c_mining_workflow", output_dir, pad_inches=0.02)
    plt.close(fig)
