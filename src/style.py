
from IPython.display import display, HTML
import pandas as pd

def setup_notebook():
    pd.set_option("display.max_columns", 100)
    pd.set_option("display.width", 180)
    pd.set_option("display.float_format", lambda x: f"{x:,.4f}")
    display(HTML("""
    <style>
      .jp-Notebook {max-width: 1500px;}
      .eda-banner {
        padding: 14px 18px; border: 1px solid #d9dee7; border-radius: 12px;
        margin: 8px 0 16px 0; background: #fafbfc;
      }
      .eda-kpi {
        display:inline-block; min-width:160px; padding:10px 14px; margin:4px;
        border:1px solid #e3e6eb; border-radius:10px; vertical-align:top;
      }
      table.dataframe {font-size: 12px;}
    </style>
    """))

def banner(title, subtitle=""):
    display(HTML(f"""
    <div class='eda-banner'>
      <div style='font-size:20px;font-weight:700'>{title}</div>
      <div style='margin-top:5px;color:#5f6670'>{subtitle}</div>
    </div>
    """))

def kpis(items):
    html = []
    for label, value in items:
        html.append(
            f"<div class='eda-kpi'><div style='color:#6b7280;font-size:12px'>{label}</div>"
            f"<div style='font-size:20px;font-weight:700;margin-top:4px'>{value}</div></div>"
        )
    display(HTML("".join(html)))
