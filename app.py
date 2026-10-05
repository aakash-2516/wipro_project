"""
Step 2 - Streamlit dashboard: Garment Productivity Predictor

Run:  streamlit run app.py
"""
import json

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils import FEATURES, add_features, estimate_units, load_clean_data

st.set_page_config(page_title="Garment Productivity Predictor",
                   page_icon="🧵", layout="wide")


# ------------------------------------------------------------------ loading
@st.cache_resource
def load_model():
    return joblib.load("models/productivity_model.joblib")


@st.cache_data
def load_all():
    data = load_clean_data()
    with open("models/metrics.json") as f:
        metrics = json.load(f)
    fi = pd.read_csv("models/feature_importance.csv")
    test_pred = pd.read_csv("models/test_predictions.csv")
    return data, metrics, fi, test_pred


try:
    model = load_model()
    data, metrics, fi, test_pred = load_all()
except FileNotFoundError:
    st.error("Model files not found. Run `python train_model.py` first.")
    st.stop()

best = metrics["results"][metrics["best_model"]]
MAE_PTS = best["test_mae"] * 100


def predict(row: dict) -> float:
    X = add_features(pd.DataFrame([row]))[FEATURES]
    return float(np.clip(model.predict(X)[0], 0.0, 1.2))


# ------------------------------------------------------------------ sidebar
st.sidebar.title("🧵 Team Inputs")

department = st.sidebar.selectbox("Department", ["sewing", "finishing"])
c1, c2 = st.sidebar.columns(2)
team = c1.selectbox("Team", [str(i) for i in range(1, 13)], index=7)
day = c2.selectbox("Day", ["Saturday", "Sunday", "Monday", "Tuesday",
                           "Wednesday", "Thursday"], index=2)
quarter = st.sidebar.selectbox("Quarter of month",
                               ["Quarter1", "Quarter2", "Quarter3", "Quarter4", "Quarter5"])

st.sidebar.markdown("**Workload**")
workers = st.sidebar.slider("Number of workers", 2, 90, 35)
target_pct = st.sidebar.slider("Targeted productivity (%)", 30, 100, 75, 5)
smv = st.sidebar.slider("SMV (standard minute value)", 2.0, 55.0, 15.0, 0.5)
wip = st.sidebar.number_input("WIP (unfinished items)", 0, 25000,
                              1000 if department == "sewing" else 0, step=50,
                              disabled=(department == "finishing"),
                              help="Work-in-progress is not tracked in the finishing department.")
style_changes = st.sidebar.selectbox("Style changes", [0, 1, 2])

st.sidebar.markdown("**Effort & interruptions**")
over_time = st.sidebar.number_input("Overtime (minutes, whole team)", 0, 26000, 4000, step=240)
incentive = st.sidebar.number_input("Incentive (BDT)", 0, 3600, 40, step=10)
idle_time = st.sidebar.number_input("Idle time (minutes)", 0, 300, 0, step=5)
idle_men = st.sidebar.number_input("Idle workers", 0, 45, 0)

row = {
    "department": department, "quarter": quarter, "day": day, "team": team,
    "targeted_productivity": target_pct / 100, "smv": smv,
    "wip": 0 if department == "finishing" else wip,
    "over_time": over_time, "incentive": incentive, "idle_time": idle_time,
    "idle_men": idle_men, "no_of_style_change": style_changes,
    "no_of_workers": workers,
}

# ------------------------------------------------------------------ predictions
pred = predict(row)
pred_pct = pred * 100
gap = pred_pct - target_pct
units = estimate_units(pred, workers, over_time, smv)
target_units = estimate_units(target_pct / 100, workers, over_time, smv)
dept_avg = data.loc[data["department"] == department, "actual_productivity"].mean() * 100

# ------------------------------------------------------------------ header
st.title("🧵 Garment Productivity Predictor")
st.caption("Trained on the UCI *Productivity Prediction of Garment Employees* dataset "
           "(1,197 team-days from a Bangladesh garment factory, Jan-Mar 2015).")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Predicted Productivity", f"{pred_pct:.1f}%", f"{gap:+.1f} pts vs target")
k2.metric("Estimated Output", f"{units:,} units", f"{units - target_units:+,} vs target output")
k3.metric(f"{department.title()} Average", f"{dept_avg:.1f}%",
          f"{pred_pct - dept_avg:+.1f} pts")
k4.metric("Typical Model Error", f"±{MAE_PTS:.1f} pts",
          help="Mean absolute error on unseen test data.", delta_color="off")

tab_pred, tab_whatif, tab_data, tab_model = st.tabs(
    ["🎯 Prediction", "🔍 What-if Analysis", "📈 Data Insights", "🧠 Model Performance"])

# ================================================================== TAB 1
with tab_pred:
    left, right = st.columns(2)
    with left:
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pred_pct, number={"suffix": "%"},
            title={"text": "Predicted vs Target"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#1f77b4"},
                "steps": [{"range": [0, 60], "color": "#f8d7da"},
                          {"range": [60, 80], "color": "#fff3cd"},
                          {"range": [80, 100], "color": "#d4edda"}],
                "threshold": {"line": {"color": "black", "width": 4},
                              "thickness": 0.8, "value": target_pct},
            }))
        gauge.update_layout(height=320, margin=dict(t=60, b=10, l=30, r=30))
        st.plotly_chart(gauge, width="stretch")
        st.caption("Black line = targeted productivity set for this team.")

    with right:
        fig = go.Figure()
        fig.add_bar(x=["Target output", "Predicted output"],
                    y=[target_units, units],
                    marker_color=["#adb5bd", "#2e86de"],
                    text=[f"{target_units:,}", f"{units:,}"], textposition="outside")
        fig.update_layout(title="Estimated Units: Target vs Predicted",
                          height=320, margin=dict(t=60, b=10), yaxis_title="Units")
        st.plotly_chart(fig, width="stretch")

    if gap >= 0:
        st.success(f"✅ Predicted to **meet or beat** the target by {gap:.1f} points.")
    elif gap >= -MAE_PTS:
        st.warning(f"⚠️ Predicted **slightly below** target ({gap:.1f} pts) - within the "
                   f"model's typical error of ±{MAE_PTS:.1f} pts, so it could go either way.")
    else:
        st.error(f"🚨 Predicted to **miss the target** by {abs(gap):.1f} points.")

    # ---- data-grounded recommendations
    tips = []
    inc_yes = data.loc[data.incentive > 0, "actual_productivity"].mean() * 100
    inc_no = data.loc[data.incentive == 0, "actual_productivity"].mean() * 100
    if incentive == 0:
        tips.append(f"No incentive set. In the data, teams with an incentive averaged "
                    f"**{inc_yes:.0f}%** vs **{inc_no:.0f}%** without one.")
    if idle_time > 0 or idle_men > 0:
        idle_yes = data.loc[data.idle_time > 0, "actual_productivity"].mean() * 100
        tips.append(f"Production interruptions recorded. Days with idle time averaged only "
                    f"**{idle_yes:.0f}%** productivity - investigate the cause.")
    if style_changes > 0:
        sc_yes = data.loc[data.no_of_style_change > 0, "actual_productivity"].mean() * 100
        sc_no = data.loc[data.no_of_style_change == 0, "actual_productivity"].mean() * 100
        tips.append(f"Style changes slow teams down: **{sc_yes:.0f}%** with changes vs "
                    f"**{sc_no:.0f}%** without.")
    if target_pct > 80:
        tips.append("The target is at the top of the range ever used (80%) - "
                    "very few teams historically hit it consistently.")
    if over_time / max(workers, 1) > 200:
        tips.append("Overtime per worker is high - check for fatigue effects.")
    if not tips:
        tips.append("No obvious red flags in these inputs.")
    with st.expander("💡 Insights for this scenario", expanded=True):
        for t in tips:
            st.write("- " + t)

    st.caption("ℹ️ *Estimated output* uses the standard efficiency formula: "
               "units = productivity × (workers × 480 min + overtime) ÷ SMV. "
               "The dataset has no unit counts, so this is an estimate.")

# ================================================================== TAB 2
with tab_whatif:
    st.write("See how the prediction changes when you vary one input and keep the rest as set in the sidebar. "
             "The red dot is your current scenario.")

    def sweep(feature, values, label, container):
        ys = [predict({**row, feature: v}) * 100 for v in values]
        fig = px.line(x=values, y=ys, markers=True,
                      labels={"x": label, "y": "Predicted productivity (%)"})
        fig.add_scatter(x=[row[feature]], y=[pred_pct], mode="markers",
                        marker=dict(size=14, color="red"), name="Current")
        fig.add_hline(y=target_pct, line_dash="dash", line_color="gray",
                      annotation_text="Target")
        fig.update_layout(height=340, margin=dict(t=20), showlegend=False)
        container.plotly_chart(fig, width="stretch")

    w1, w2, w3, w4 = st.tabs(["Workers", "Incentive", "Overtime", "Style changes"])
    sweep("no_of_workers", list(range(5, 91, 5)), "Number of workers", w1)
    sweep("incentive", [0, 10, 20, 30, 40, 50, 60, 70, 80, 100, 150, 200], "Incentive (BDT)", w2)
    sweep("over_time", list(range(0, 15001, 1000)), "Overtime (minutes)", w3)
    sweep("no_of_style_change", [0, 1, 2], "Style changes", w4)

    st.caption("Note: the model learns associations in historical data, not guaranteed cause-and-effect. "
               "Use these curves as guidance, not proof.")

# ================================================================== TAB 3
with tab_data:
    a, b = st.columns(2)
    with a:
        fig = px.histogram(data, x="actual_productivity", nbins=40, color="department",
                           barmode="overlay", opacity=0.7,
                           title="Distribution of actual productivity",
                           labels={"actual_productivity": "Actual productivity"})
        fig.update_layout(height=340)
        st.plotly_chart(fig, width="stretch")
    with b:
        by_day = (data.groupby(["day", "department"])["actual_productivity"].mean()
                  .reset_index())
        order = ["Saturday", "Sunday", "Monday", "Tuesday", "Wednesday", "Thursday"]
        fig = px.bar(by_day, x="day", y="actual_productivity", color="department",
                     barmode="group", category_orders={"day": order},
                     title="Average productivity by weekday",
                     labels={"actual_productivity": "Avg productivity"})
        fig.update_layout(height=340)
        st.plotly_chart(fig, width="stretch")

    c, d = st.columns(2)
    with c:
        daily = data.groupby("date")["actual_productivity"].mean().reset_index()
        fig = px.line(daily, x="date", y="actual_productivity",
                      title="Daily average productivity (Jan-Mar 2015)")
        fig.update_layout(height=340)
        st.plotly_chart(fig, width="stretch")
    with d:
        by_team = (data.groupby("team")["actual_productivity"].mean().reset_index()
                   .assign(team=lambda t: t["team"].astype(int)).sort_values("team"))
        fig = px.bar(by_team, x="team", y="actual_productivity",
                     title="Average productivity by team",
                     labels={"actual_productivity": "Avg productivity"})
        fig.update_layout(height=340, xaxis=dict(dtick=1))
        st.plotly_chart(fig, width="stretch")

    num_cols = ["targeted_productivity", "smv", "wip", "over_time", "incentive",
                "idle_time", "idle_men", "no_of_style_change", "no_of_workers",
                "actual_productivity"]
    corr = data[num_cols].corr().round(2)
    fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r",
                    zmin=-1, zmax=1, title="Correlation matrix")
    fig.update_layout(height=520)
    st.plotly_chart(fig, width="stretch")

    with st.expander("Browse the dataset"):
        st.dataframe(data.drop(columns=["overtime_per_worker", "has_incentive"]),
                     width="stretch")

# ================================================================== TAB 4
with tab_model:
    st.write(f"**Selected model:** {metrics['best_model']} "
             f"(trained on {metrics['train_rows']} rows, tested on {metrics['test_rows']} unseen rows)")

    res = (pd.DataFrame(metrics["results"]).T.reset_index()
           .rename(columns={"index": "Model", "cv_r2": "CV R²", "test_r2": "Test R²",
                            "test_mae": "Test MAE", "test_rmse": "Test RMSE"}))
    st.dataframe(res, hide_index=True, width="stretch")

    m1, m2 = st.columns(2)
    with m1:
        fig = px.scatter(test_pred, x="actual", y="predicted", color="department",
                         opacity=0.7, title="Actual vs Predicted (test set)")
        fig.add_shape(type="line", x0=0.2, y0=0.2, x1=1.1, y1=1.1,
                      line=dict(dash="dash", color="gray"))
        fig.update_layout(height=400)
        st.plotly_chart(fig, width="stretch")
    with m2:
        fig = px.bar(fi.head(12).sort_values("importance"), x="importance", y="feature",
                     orientation="h", title="What drives productivity? (permutation importance)")
        fig.update_layout(height=400)
        st.plotly_chart(fig, width="stretch")

    st.info(
        f"**How to read this:** R² of {best['test_r2']:.2f} means the model explains roughly "
        f"{best['test_r2'] * 100:.0f}% of the variation in productivity, with a typical error of "
        f"±{MAE_PTS:.1f} percentage points. That is typical for this dataset - it is small (1,197 rows) and "
        "productivity also depends on factors that were not recorded (worker skill, machine breakdowns, "
        "material delays).")
