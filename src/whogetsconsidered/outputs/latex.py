"""LaTeX rendering helpers for research tables and appendix material."""

from __future__ import annotations

from pathlib import Path

import polars as pl
from jinja2 import Template


LATEX_TEMPLATE = Template(
    r"""
\begin{table}[!htbp]
\centering
\caption{{ caption }}
\label{ {{ label }} }
\begin{tabular}{ {{ alignment }} }
\hline
{% for header in headers -%}
{{ header }}{% if not loop.last %} & {% endif %}
{% endfor %} \\
\hline
{% for row in rows -%}
{% for cell in row -%}
{{ cell }}{% if not loop.last %} & {% endif %}
{% endfor %} \\
{% endfor %}
\hline
\end{tabular}
\end{table}
""".strip()
)


def render_latex_table(df: pl.DataFrame, *, caption: str, label: str) -> str:
    """Render a small Polars dataframe as a basic LaTeX tabular table."""

    return LATEX_TEMPLATE.render(
        caption=caption,
        label=label,
        alignment="l" * max(1, len(df.columns)),
        headers=df.columns,
        rows=df.rows(),
    )


def write_latex_table(path: str | Path, df: pl.DataFrame, *, caption: str, label: str) -> Path:
    """Write a rendered LaTeX table to disk."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_latex_table(df, caption=caption, label=label), encoding="utf-8")
    return output_path
