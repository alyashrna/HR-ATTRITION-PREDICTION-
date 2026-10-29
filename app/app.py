"""
Employee Attrition Risk — HR analytics dashboard.

    streamlit run app/app.py
"""
import json
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from features import CATEGORICAL, NUMERIC, add_features  # noqa: E402

MODEL_PATH = ROOT / "models" / "attrition_model.joblib"
REPORTS = ROOT / "reports"

# ----------------------------------------------------------------- design tokens
NAVY = "#1F3A68"
INK = "#1B2333"
SLATE = "#5B6475"
LINE = "#E3E7EE"
ACCENT = "#2A78D6"
SOFT = "#C9D6EA"
RISK = {"High": "#C8423B", "Medium": "#D9922B", "Low": "#2E8B62"}
FONT = "Inter, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"

st.set_page_config(page_title="Attrition Risk | HR Analytics", page_icon="▣", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"], .stMarkdown, .stText, button, input, select, textarea {{ font-family: {FONT}; }}
#MainMenu, footer, [data-testid="stDecoration"] {{ visibility: hidden; height: 0; }}
header[data-testid="stHeader"] {{ background: transparent; }}
.block-container {{ padding-top: 1.6rem; padding-bottom: 2.5rem; max-width: 1400px; }}

/* sidebar */
section[data-testid="stSidebar"] {{ background: {NAVY}; }}
section[data-testid="stSidebar"] * {{ color: #E8EDF5 !important; }}
section[data-testid="stSidebar"] [data-baseweb="select"] * {{ color: {INK} !important; }}
section[data-testid="stSidebar"] [data-baseweb="tag"] {{ background: #DCE5F3 !important; }}
section[data-testid="stSidebar"] hr {{ border-color: rgba(255,255,255,.15); }}
.side-brand {{ font-size: 1.05rem; font-weight: 700; letter-spacing: .02em; margin-bottom: .1rem; }}
.side-sub {{ font-size: .78rem; opacity: .7; margin-bottom: 1.2rem; }}
.side-label {{ font-size: .7rem; text-transform: uppercase; letter-spacing: .12em; opacity: .6;
               margin: 1rem 0 .35rem; }}

/* page header */
.page-head {{ background: linear-gradient(90deg, {NAVY} 0%, #2B4F8A 100%); color: #fff;
              border-radius: 10px; padding: 22px 28px; margin-bottom: 18px; }}
.page-head .eyebrow {{ font-size: .72rem; letter-spacing: .14em; text-transform: uppercase; opacity: .75; }}
.page-head h1 {{ color: #fff; font-size: 1.65rem; font-weight: 700; margin: .2rem 0 .25rem; padding: 0; }}
.page-head p {{ margin: 0; font-size: .9rem; opacity: .85; }}

/* KPI cards */
.kpi {{ background: #fff; border: 1px solid {LINE}; border-radius: 10px; padding: 16px 18px;
        border-top: 3px solid {NAVY}; min-height: 150px; }}
.kpi .value.small {{ font-size: 1.25rem; padding-top: .3rem; }}
.kpi.alert {{ border-top-color: {RISK['High']}; }}
.kpi .label {{ font-size: .72rem; color: {SLATE}; text-transform: uppercase; letter-spacing: .08em;
               font-weight: 600; }}
.kpi .value {{ font-size: 1.75rem; font-weight: 700; color: {INK}; margin: .25rem 0 .1rem;
               font-variant-numeric: tabular-nums; }}
.kpi .note {{ font-size: .78rem; color: {SLATE}; }}

/* section titles & cards */
.section {{ font-size: .95rem; font-weight: 700; color: {INK}; margin: 8px 0 2px; }}
.section-sub {{ font-size: .8rem; color: {SLATE}; margin-bottom: 8px; }}
div[data-testid="stVerticalBlockBorderWrapper"] {{ background: #fff; border-color: {LINE} !important;
        border-radius: 10px; }}
.insight {{ border-left: 3px solid {NAVY}; padding: 8px 12px; margin-bottom: 10px; background: #F7F9FC;
            border-radius: 0 6px 6px 0; font-size: .86rem; color: {INK}; }}
.insight b {{ color: {NAVY}; }}
.badge {{ display: inline-block; padding: 3px 12px; border-radius: 999px; font-size: .8rem;
          font-weight: 600; color: #fff; }}
.foot {{ font-size: .75rem; color: {SLATE}; border-top: 1px solid {LINE}; margin-top: 28px; padding-top: 10px; }}
div[data-testid="stDataFrame"] {{ border: 1px solid {LINE}; border-radius: 8px; }}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------- data/model
if not MODEL_PATH.exists():  # first run (e.g. Streamlit Community Cloud)
    with st.spinner("Preparing data and training the model for first use (about 30 seconds)..."):
        subprocess.run([sys.executable, str(ROOT / "run_pipeline.py")], check=True, cwd=ROOT)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_data():
    emp = pd.read_csv(ROOT / "data" / "employees.csv")
    risk = pd.read_csv(REPORTS / "current_employee_risk.csv")
    metrics = json.loads((REPORTS / "metrics.json").read_text())
    comparison = pd.read_csv(REPORTS / "model_comparison.csv")
    importance = pd.read_csv(REPORTS / "feature_importance.csv")
    return emp, risk, metrics, comparison, importance


bundle = load_model()
model, peer, threshold = bundle["model"], bundle["peer_medians"], bundle["threshold"]
emp, risk, metrics, comparison, importance = load_data()
emp["left"] = emp["attrition"].eq("Yes")


# ---------------------------------------------------------------------- helpers
def rm(v: float) -> str:
    if v >= 1e6:
        return f"RM {v / 1e6:,.2f}M"
    if v >= 1e3:
        return f"RM {v / 1e3:,.0f}K"
    return f"RM {v:,.0f}"


def kpi(col, label, value, note="", alert=False, small=False):
    col.markdown(f'<div class="kpi {"alert" if alert else ""}"><div class="label">{label}</div>'
                 f'<div class="value {"small" if small else ""}">{value}</div><div class="note">{note}</div></div>',
                 unsafe_allow_html=True)


def header(eyebrow, title, sub):
    st.markdown(f'<div class="page-head"><div class="eyebrow">{eyebrow}</div><h1>{title}</h1>'
                f'<p>{sub}</p></div>', unsafe_allow_html=True)


def section(title, sub=""):
    st.markdown(f'<div class="section">{title}</div>' + (f'<div class="section-sub">{sub}</div>' if sub else ""),
                unsafe_allow_html=True)


def layout(fig, h=320, legend=False):
    fig.update_layout(height=h, margin=dict(l=8, r=16, t=8, b=8), showlegend=legend,
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                      font=dict(family=FONT, size=12, color=SLATE),
                      hoverlabel=dict(font_family=FONT, bgcolor="white", bordercolor=LINE),
                      legend=dict(orientation="h", y=1.08, x=0, title=None))
    fig.update_xaxes(showgrid=False, linecolor=LINE, ticks="")
    fig.update_yaxes(gridcolor="#EEF1F5", zeroline=False, linecolor=LINE)
    return fig


def hbar(labels, values, highlight=None, fmt="{:.1f}%", ref=None, h=300, colors=None):
    labels, values = list(labels), list(values)
    if colors is None:
        colors = [NAVY if (highlight is not None and v >= highlight) else SOFT for v in values]
    fig = go.Figure(go.Bar(x=values, y=labels, orientation="h", marker_color=colors,
                           marker_line_width=0, text=[fmt.format(v) for v in values],
                           textposition="outside", textfont=dict(color=INK, size=11),
                           hovertemplate="%{y}: %{x:.1f}<extra></extra>", cliponaxis=False))
    if ref is not None:
        fig.add_vline(x=ref, line_dash="dot", line_color=SLATE, line_width=1,
                      annotation_text=f"Company avg {ref:.1f}%", annotation_position="top",
                      annotation_font=dict(size=10, color=SLATE))
    fig.update_xaxes(showticklabels=False, range=[0, max(values) * 1.22])
    fig.update_yaxes(showgrid=False)
    fig = layout(fig, h)
    if ref is not None:
        fig.update_layout(margin=dict(t=28))
    return fig


def score(row: pd.DataFrame) -> float:
    feats, _ = add_features(row, peer)
    return float(model.predict_proba(feats[NUMERIC + CATEGORICAL])[:, 1][0])


def band_of(p: float) -> str:
    return "High" if p >= threshold else "Medium" if p >= 0.3 else "Low"


def explain(row: pd.DataFrame) -> pd.Series:
    """Per-feature contribution to the log-odds (logistic regression)."""
    feats, _ = add_features(row, peer)
    prep, clf = model.named_steps["prep"], model.named_steps["model"]
    if not hasattr(clf, "coef_"):
        return pd.Series(dtype=float)
    x = prep.transform(feats[NUMERIC + CATEGORICAL])
    x = x.toarray() if hasattr(x, "toarray") else x
    contrib = pd.Series(x[0] * clf.coef_[0], index=prep.get_feature_names_out())
    orig = {}
    for n in contrib.index:
        b = n.split("__", 1)[1]
        orig[n] = next((c for c in CATEGORICAL if b.startswith(c + "_")), b)
    return contrib.groupby(orig).sum().sort_values(ascending=False)


PRETTY = {"overtime": "Regular overtime", "pay_vs_peer_median": "Pay vs. peer median",
          "is_new_joiner": "New joiner (≤ 1 year)", "work_mode": "Work arrangement",
          "work_life_balance": "Work-life balance", "job_satisfaction": "Job satisfaction",
          "environment_satisfaction": "Environment satisfaction", "num_companies_worked": "Previous employers",
          "manager_changes": "Manager changes", "total_working_years": "Total working years",
          "monthly_income": "Monthly income", "job_level": "Job level", "department": "Department",
          "distance_from_home_km": "Commute distance", "years_since_last_promotion": "Years since promotion",
          "last_salary_hike_pct": "Last salary increment", "marital_status": "Marital status",
          "years_at_company": "Tenure", "age": "Age", "training_hours_last_year": "Training hours",
          "promotion_stagnation": "Promotion stagnation", "job_hopper_index": "Job-change frequency",
          "performance_rating": "Performance rating", "education": "Education", "gender": "Gender"}

# ---------------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown('<div class="side-brand">HR ANALYTICS</div>'
                '<div class="side-sub">Workforce Attrition Risk</div>', unsafe_allow_html=True)
    st.markdown('<div class="side-label">Navigation</div>', unsafe_allow_html=True)
    page = st.radio("Navigation", ["Executive Overview", "Risk Register", "Employee Assessment",
                                   "Model & Methodology"], label_visibility="collapsed")
    st.markdown('<div class="side-label">Filters</div>', unsafe_allow_html=True)
    f_dept = st.multiselect("Department", sorted(emp["department"].unique()), placeholder="All departments")
    f_level = st.multiselect("Job level", [1, 2, 3, 4, 5], placeholder="All levels")
    st.markdown("---")
    st.markdown(f'<div class="side-sub">Model: {bundle["model_name"]}<br>'
                f'High-risk threshold: {threshold:.2f}<br>Data: synthetic, {len(emp):,} employees</div>',
                unsafe_allow_html=True)


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    if f_dept:
        df = df[df["department"].isin(f_dept)]
    if f_level:
        df = df[df["job_level"].isin(f_level)]
    return df


E = apply_filters(emp)
R = apply_filters(risk)
scope = ", ".join(f_dept) if f_dept else "All departments"
if f_level:
    scope += " · Level " + ", ".join(map(str, f_level))

if E.empty:
    st.warning("No employees match the selected filters.")
    st.stop()

# ============================================================ EXECUTIVE OVERVIEW
if page == "Executive Overview":
    header("Executive Overview", "Workforce Attrition Risk", f"Scope: {scope}")

    rate, co_rate = E["left"].mean() * 100, emp["left"].mean() * 100
    high = R[R["risk_band"] == "High"]
    exposure = (high["monthly_income"] * 6).sum()
    c = st.columns(5)
    kpi(c[0], "Headcount", f"{len(E):,}", "Employees in scope")
    kpi(c[1], "Attrition rate", f"{rate:.1f}%",
        f"{rate - co_rate:+.1f} pts vs company" if (f_dept or f_level) else "Last 12 months")
    kpi(c[2], "Leavers", f"{int(E['left'].sum()):,}", "Last 12 months")
    kpi(c[3], "High-risk employees", f"{len(high):,}",
        f"{len(high) / max(len(R), 1):.0%} of current staff", alert=True)
    kpi(c[4], "Cost exposure", rm(exposure), "High-risk replacement cost", alert=True)

    st.write("")
    left, right = st.columns([3, 2], gap="medium")
    with left:
        with st.container(border=True):
            section("Attrition rate by department", "Navy = above company average")
            d = E.groupby("department")["left"].mean().mul(100).sort_values()
            st.plotly_chart(hbar(d.index, d.values, highlight=co_rate, ref=co_rate, h=310),
                            width="stretch", config={"displayModeBar": False})
    with right:
        with st.container(border=True):
            section("Key findings")
            ot = E.groupby("overtime")["left"].mean() * 100
            pay = E.assign(r=lambda x: add_features(x, peer)[0]["pay_vs_peer_median"])
            below = pay.loc[pay["r"] < 0.9, "left"].mean() * 100
            above = pay.loc[pay["r"] > 1.1, "left"].mean() * 100
            wm = E.groupby("work_mode")["left"].mean() * 100
            nj = E.loc[E["years_at_company"] <= 1, "left"].mean() * 100
            for html in [
                f"<b>Overtime</b>: employees on regular overtime leave at {ot.get('Yes', 0):.0f}%, "
                f"compared with {ot.get('No', 0):.0f}% for others.",
                f"<b>Pay competitiveness</b>: staff paid below 90% of their peer median leave at "
                f"{below:.0f}%, against {above:.0f}% for those paid above 110%.",
                f"<b>Work arrangement</b>: onsite attrition is {wm.get('Onsite', 0):.0f}%, "
                f"compared with {wm.get('Hybrid', 0):.0f}% for hybrid.",
                f"<b>Early tenure</b>: {nj:.0f}% of employees in their first year leave.",
            ]:
                st.markdown(f'<div class="insight">{html}</div>', unsafe_allow_html=True)

    st.write("")
    cols = st.columns(3, gap="medium")
    feats_E = add_features(E, peer)[0]
    feats_E["pay_band"] = pd.cut(feats_E["pay_vs_peer_median"], [0, 0.9, 1.1, 10],
                                 labels=["Below peers", "In line", "Above peers"])
    feats_E["tenure_band"] = pd.cut(feats_E["years_at_company"], [-1, 1, 3, 5, 50],
                                    labels=["0–1 yrs", "2–3 yrs", "4–5 yrs", "6+ yrs"])
    for col, (field, title) in zip(cols, [("overtime", "By overtime"), ("pay_band", "By pay vs. peers"),
                                          ("tenure_band", "By tenure")]):
        with col, st.container(border=True):
            section(title)
            g = feats_E.groupby(field, observed=True)["left"].mean().mul(100)
            st.plotly_chart(hbar(g.index.astype(str), g.values, highlight=co_rate, h=200),
                            width="stretch", config={"displayModeBar": False})

# ================================================================= RISK REGISTER
elif page == "Risk Register":
    header("Risk Register", "Current Employees at Risk",
           f"Model-scored list of current employees · High risk = score ≥ {threshold:.2f}")

    c = st.columns(4)
    counts = R["risk_band"].value_counts()
    kpi(c[0], "Current employees", f"{len(R):,}", scope)
    kpi(c[1], "High risk", f"{counts.get('High', 0):,}", "Priority for retention action", alert=True)
    kpi(c[2], "Medium risk", f"{counts.get('Medium', 0):,}", "Monitor quarterly")
    kpi(c[3], "Monthly payroll at high risk",
        rm(R.loc[R["risk_band"] == "High", "monthly_income"].sum()), "Sum of monthly salaries")

    st.write("")
    left, right = st.columns([1, 2], gap="medium")
    with left:
        with st.container(border=True):
            section("High-risk employees by department")
            h = R[R["risk_band"] == "High"].groupby("department").size().sort_values()
            if not h.empty:
                st.plotly_chart(hbar(h.index, h.values, fmt="{:.0f}", h=320, colors=[NAVY] * len(h)),
                                width="stretch", config={"displayModeBar": False})
    with right:
        with st.container(border=True):
            section("Employee register", "Sorted by risk score. Filter and export for follow-up.")
            bands = st.segmented_control("Risk band", ["High", "Medium", "Low"], default=["High"],
                                         selection_mode="multi", label_visibility="collapsed")
            view = (R[R["risk_band"].isin(bands)] if bands else R).copy()
            view["pay_vs_peer_median"] = view["pay_vs_peer_median"] * 100
            view = view.rename(columns={
                "employee_id": "Employee ID", "department": "Department", "job_level": "Level",
                "monthly_income": "Monthly income (RM)", "overtime": "Overtime",
                "job_satisfaction": "Job satisfaction", "years_since_last_promotion": "Yrs since promotion",
                "pay_vs_peer_median": "Pay vs. peers", "attrition_risk": "Risk score",
                "risk_band": "Risk band"})
            view = view[["Employee ID", "Department", "Risk score", "Risk band", "Level",
                         "Monthly income (RM)", "Pay vs. peers", "Overtime", "Job satisfaction",
                         "Yrs since promotion"]]
            st.dataframe(
                view, hide_index=True, width="stretch", height=330,
                column_config={
                    "Risk score": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f"),
                    "Monthly income (RM)": st.column_config.NumberColumn(format="%,d"),
                    "Pay vs. peers": st.column_config.NumberColumn(format="%.0f%%",
                                                                   help="Salary ÷ median of same department & level"),
                })
            st.download_button("Export register (CSV)", view.to_csv(index=False).encode(),
                               "attrition_risk_register.csv", mime="text/csv")
    st.caption("Pay vs. peers: 100% = paid exactly at the median for the same department and job level.")

# ============================================================ EMPLOYEE ASSESSMENT
elif page == "Employee Assessment":
    header("Employee Assessment", "Individual Risk Assessment",
           "Enter an employee profile to estimate attrition risk, identify drivers and test interventions")

    with st.container(border=True):
        section("Employee profile")
        c1, c2, c3 = st.columns(3, gap="large")
        with c1:
            st.markdown("**Role & compensation**")
            dept = st.selectbox("Department", sorted(emp["department"].unique()),
                                index=sorted(emp["department"].unique()).index("Operations"))
            level = st.slider("Job level", 1, 5, 2)
            income = st.number_input("Monthly income (RM)", 1500, 40000, 4200, step=100)
            hike = st.slider("Last salary increment (%)", 0.0, 15.0, 3.0, 0.5)
        with c2:
            st.markdown("**Working conditions**")
            ot = st.radio("Regular overtime", ["Yes", "No"], horizontal=True)
            wm = st.radio("Work arrangement", ["Onsite", "Hybrid", "Remote"], horizontal=True)
            dist = st.slider("Commute distance (km)", 1, 80, 22)
            js = st.select_slider("Job satisfaction", [1, 2, 3, 4], 2,
                                  format_func=lambda v: {1: "1 – Low", 2: "2", 3: "3", 4: "4 – High"}[v])
            es = st.select_slider("Environment satisfaction", [1, 2, 3, 4], 2)
            wlb = st.select_slider("Work-life balance", [1, 2, 3, 4], 2)
        with c3:
            st.markdown("**Career history**")
            age = st.slider("Age", 21, 60, 27)
            yac = st.slider("Years at company", 0, 30, 1)
            ysp = st.slider("Years since last promotion", 0, 15, 1)
            twy = st.slider("Total working years", 0, 40, 4)
            ncw = st.slider("Previous employers", 0, 9, 2)

    emp_row = dict(department=dept, job_level=level, monthly_income=income, last_salary_hike_pct=hike,
                   overtime=ot, work_mode=wm, distance_from_home_km=dist, job_satisfaction=js,
                   environment_satisfaction=es, work_life_balance=wlb, age=age, years_at_company=yac,
                   years_since_last_promotion=ysp, total_working_years=twy, num_companies_worked=ncw,
                   gender="Female", marital_status="Single", education="Bachelor's", manager_changes=1,
                   performance_rating=3, training_hours_last_year=20)
    row = pd.DataFrame([emp_row])
    p = score(row)
    band = band_of(p)

    st.write("")
    g1, g2 = st.columns([2, 3], gap="medium")
    with g1, st.container(border=True):
        section("Risk score")
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=p, number=dict(valueformat=".2f", font=dict(size=40, color=INK)),
            gauge=dict(axis=dict(range=[0, 1], tickwidth=0, tickcolor=LINE, tickfont=dict(size=10)),
                       bar=dict(color=INK, thickness=0.22), bgcolor="rgba(0,0,0,0)", borderwidth=0,
                       steps=[dict(range=[0, 0.3], color="#DCEFE5"),
                              dict(range=[0.3, threshold], color="#F6E6CC"),
                              dict(range=[threshold, 1], color="#F3D6D4")],
                       threshold=dict(line=dict(color=RISK["High"], width=3), thickness=0.85,
                                      value=threshold))))
        fig.update_layout(height=230, margin=dict(l=25, r=25, t=10, b=0), paper_bgcolor="rgba(0,0,0,0)",
                          font=dict(family=FONT))
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        action = {"High": "Recommend a retention conversation with the line manager within 2 weeks.",
                  "Medium": "Monitor and review at the next quarterly check-in.",
                  "Low": "No action required."}[band]
        st.markdown(f'<div style="text-align:center"><span class="badge" style="background:{RISK[band]}">'
                    f'{band.upper()} RISK</span><div class="section-sub" style="margin-top:8px">{action}</div></div>',
                    unsafe_allow_html=True)

    with g2, st.container(border=True):
        section("Main risk drivers for this employee", "Contribution relative to an average employee")
        ex = explain(row)
        if not ex.empty:
            ex = pd.concat([ex.head(5), ex.tail(3)]).iloc[::-1]
            fig = go.Figure(go.Bar(
                x=ex.values, y=[PRETTY.get(k, k) for k in ex.index], orientation="h",
                marker_color=[RISK["High"] if v > 0 else ACCENT for v in ex.values],
                hovertemplate="%{y}: %{x:+.2f}<extra></extra>"))
            fig.add_vline(x=0, line_color=SLATE, line_width=1)
            fig.update_xaxes(title=dict(text="← lowers risk      raises risk →", font=dict(size=11)),
                             zeroline=False, showticklabels=False)
            st.plotly_chart(layout(fig, 290), width="stretch", config={"displayModeBar": False})

    st.write("")
    with st.container(border=True):
        section("Intervention scenarios", "Estimated risk score if HR takes each action")
        new_level = min(level + 1, 5)
        promo_pay = max(income * 1.1, float(peer.get((dept, new_level), 0)))
        scenarios = {
            "Current": {},
            "Remove overtime": {"overtime": "No"},
            "Salary +15%": {"monthly_income": income * 1.15},
            "Move to hybrid": {"work_mode": "Hybrid"},
            "Promote one level": {"job_level": new_level, "years_since_last_promotion": 0,
                                  "monthly_income": promo_pay},
            "Combined package": {"overtime": "No", "work_mode": "Hybrid", "job_level": new_level,
                                 "years_since_last_promotion": 0, "monthly_income": promo_pay},
        }
        res = pd.DataFrame([{"Scenario": k, "Risk": score(row.assign(**v))} for k, v in scenarios.items()])
        res["Band"] = res["Risk"].map(band_of)
        res["Change"] = res["Risk"] - p
        fig = go.Figure(go.Bar(x=res["Scenario"], y=res["Risk"], marker_color=[RISK[b] for b in res["Band"]],
                               text=res["Risk"].map("{:.2f}".format), textposition="outside",
                               textfont=dict(color=INK), width=0.55, cliponaxis=False,
                               hovertemplate="%{x}: %{y:.2f}<extra></extra>"))
        fig.add_hline(y=threshold, line_dash="dot", line_color=RISK["High"], line_width=1.5,
                      annotation_text="High-risk threshold", annotation_position="top right",
                      annotation_font=dict(size=10, color=RISK["High"]))
        fig.update_yaxes(range=[0, 1.08], title=None, tickformat=".1f")
        st.plotly_chart(layout(fig, 300), width="stretch", config={"displayModeBar": False})
        best = res.iloc[1:].sort_values("Risk").iloc[0]
        st.markdown(f'<div class="insight"><b>Most effective action:</b> {best["Scenario"]} reduces the risk score '
                    f'from {p:.2f} to {best["Risk"]:.2f} ({best["Change"]:+.2f}).</div>', unsafe_allow_html=True)

# ============================================================ MODEL & METHODOLOGY
else:
    header("Model & Methodology", "Model Performance and Assumptions",
           "How the risk scores are produced and how reliable they are")
    m = metrics
    c = st.columns(5)
    kpi(c[0], "Model", m["best_model"], "Selected by 5-fold CV", small=True)
    kpi(c[1], "ROC-AUC", f"{m['test_roc_auc']:.2f}", "Held-out test set")
    kpi(c[2], "PR-AUC", f"{m['test_pr_auc']:.2f}", f"Random baseline {m['baseline_pr_auc (attrition rate)']:.2f}")
    kpi(c[3], "Recall / Precision", f"{m['recall']:.0%} / {m['precision']:.0%}", f"At threshold {m['chosen_threshold']}")
    kpi(c[4], "Est. net savings", rm(m["est_net_savings_whole_workforce_RM"]), "Applied to whole workforce")

    st.write("")
    left, mid, right = st.columns([1.1, 1, 1.2], gap="medium")
    with left, st.container(border=True):
        section("Model comparison", "5-fold cross-validated ROC-AUC")
        comp = comparison.sort_values("cv_roc_auc_mean")
        st.plotly_chart(hbar(comp["model"], comp["cv_roc_auc_mean"], fmt="{:.3f}", h=200,
                             colors=[NAVY if mname == m["best_model"] else SOFT for mname in comp["model"]]),
                        width="stretch", config={"displayModeBar": False})
    with mid, st.container(border=True):
        section("Confusion matrix", "Test set, 800 employees")
        cm = m["confusion_matrix"]
        z = [[cm["TN"], cm["FP"]], [cm["FN"], cm["TP"]]]
        fig = go.Figure(go.Heatmap(z=z, x=["Predicted<br>stay", "Predicted<br>leave"],
                                   y=["Actual stayed", "Actual left"], colorscale=[[0, "#EEF2F8"], [1, NAVY]],
                                   showscale=False, text=z, texttemplate="%{text}",
                                   textfont=dict(size=16), hoverinfo="skip"))
        fig.update_yaxes(autorange="reversed", showgrid=False)
        fig.update_xaxes(tickangle=0, side="top")
        st.plotly_chart(layout(fig, 200), width="stretch", config={"displayModeBar": False})
    with right, st.container(border=True):
        section("Top predictors", "Permutation importance (drop in ROC-AUC)")
        top = importance.head(6).iloc[::-1]
        st.plotly_chart(hbar([PRETTY.get(f, f) for f in top["feature"]], top["importance"], fmt="{:.3f}",
                             h=200, colors=[NAVY] * len(top)),
                        width="stretch", config={"displayModeBar": False})

    st.write("")
    with st.container(border=True):
        section("Methodology & assumptions")
        a, b = st.columns(2, gap="large")
        a.markdown("""
- **Data:** 4,000 employees, binary label *left within the last 12 months* (18.1% attrition)
- **Validation:** stratified 80/20 split, 5-fold cross-validation for model selection
- **Leakage control:** peer-median salaries learned on training data only
- **Imbalance:** class-weighted models, evaluated with PR-AUC rather than accuracy
""")
        b.markdown(f"""
- **Threshold:** {m['chosen_threshold']} chosen to maximise estimated net savings
- **Replacement cost:** 6 months' salary per leaver
- **Intervention:** RM3,000 per flagged employee, 30% success rate
- **Use:** scores rank employees for a manager conversation; they must not drive automated decisions
""")

st.markdown('<div class="foot">HR Analytics · Workforce Attrition Risk · Synthetic dataset for portfolio '
            'demonstration; figures do not represent any real organisation.</div>', unsafe_allow_html=True)
