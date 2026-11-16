"""Flight Delay Risk Checker - standalone Streamlit app.
Needs only: this file + flight_delay_bundle.joblib (same folder) + requirements.txt
Run:  streamlit run streamlit_app.py
"""
import datetime as dt
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Flight Delay Risk Checker", page_icon="✈️")


@st.cache_resource
def load_bundle():
    return joblib.load(Path(__file__).parent / "flight_delay_bundle.joblib")


B = load_bundle()
prior, tables = B["prior"], B["rate_tables"]


def make_features(airline, origin, dest, date, dep_min, sched_time, distance, tz_shift):
    """Same feature logic as the notebook (Stage 4)."""
    arr_min = (dep_min + sched_time + tz_shift) % 1440
    dep_hour, arr_hour = dep_min // 60, arr_min // 60
    route = f"{origin}_{dest}"
    row = {
        "MONTH": date.month, "DAY_OF_WEEK": date.weekday() + 1, "DEP_HOUR": dep_hour, "ARR_HOUR": arr_hour,
        "SCHEDULED_TIME": sched_time, "DISTANCE": distance, "IS_WEEKEND": int(date.weekday() >= 5),
        "AIRLINE_RATE": tables["AIRLINE_RATE"].get(airline, prior),
        "ORIGIN_RATE": tables["ORIGIN_RATE"].get(origin, prior),
        "DEST_RATE": tables["DEST_RATE"].get(dest, prior),
        "ROUTE_RATE": tables["ROUTE_RATE"].get(route, prior),
        "ORIGIN_HOUR_RATE": tables["ORIGIN_HOUR_RATE"].get(f"{origin}_{dep_hour}", prior),
    }
    return pd.DataFrame([row])[B["features"]]


def predict(*args):
    return float(B["model"].predict_proba(make_features(*args))[0, 1])


st.title("✈️ Flight Delay Risk Checker")
st.caption("Probability that a US domestic flight arrives **more than 15 minutes late**, estimated before departure.")

airports, airlines, routes = B["airports"], B["airlines"], B["routes"]
with st.sidebar:
    st.header("Flight details")
    origins = sorted({r.split("_")[0] for r in routes}, key=lambda c: airports.get(c, c))
    origin = st.selectbox("Origin", origins, format_func=lambda c: f"{c} - {airports.get(c, c)}")
    dests = sorted([r.split("_")[1] for r in routes if r.startswith(origin + "_")], key=lambda c: airports.get(c, c))
    dest = st.selectbox("Destination", dests, format_func=lambda c: f"{c} - {airports.get(c, c)}")
    airline = st.selectbox("Airline", sorted(airlines, key=airlines.get), format_func=lambda c: f"{airlines[c]} ({c})")
    date = st.date_input("Flight date", dt.date.today() + dt.timedelta(days=7))
    dep_time = st.time_input("Scheduled departure (local)", dt.time(17, 0), step=300)

info = routes[f"{origin}_{dest}"]
distance, sched_time, tz_shift = info["distance"], info["sched_time"], info["tz_shift"]
dep_min = dep_time.hour * 60 + dep_time.minute

p = predict(airline, origin, dest, date, dep_min, sched_time, distance, tz_shift)
base, thr = B["prior"], B["threshold"]
level = "Low" if p < 0.8 * base else ("Medium" if p < thr else "High")
colour = {"Low": "green", "Medium": "orange", "High": "red"}[level]

c1, c2 = st.columns(2)
c1.metric("Probability of arriving > 15 min late", f"{p:.0%}", f"{(p - base) * 100:+.0f} pts vs average ({base:.0%})", delta_color="inverse")
c2.markdown(f"### Risk level\n## :{colour}[{level}]")
st.progress(min(max(p, 0.0), 1.0))
st.caption(f"Route {origin} → {dest}: {distance:,.0f} miles, scheduled duration {sched_time:.0f} min.")

st.subheader("Risk by departure hour (same flight, same date)")
hours = pd.Series({h: predict(airline, origin, dest, date, h * 60, sched_time, distance, tz_shift) for h in range(24)})
st.line_chart(hours.rename("delay probability"))
st.caption(f"Lowest-risk departure hour: **{hours.idxmin():02d}:00** ({hours.min():.0%}).")

st.subheader("What drives delays (model-wide)")
d = pd.DataFrame(B["drivers"]).set_index("factor")["auc_drop"]
st.bar_chart(d, horizontal=True)

m = B["metrics"]
with st.expander("About the model"):
    st.write(f"Model: **{m['model']}** · test ROC-AUC **{m['roc_auc']:.3f}** · PR-AUC {m['pr_auc']:.3f} (base rate {m['base_rate']:.2f}) · "
             f"recall {m['recall']:.0%}, precision {m['precision']:.0%} at threshold {thr:.2f} · "
             f"the riskiest 20% of flights contain {m['top20_capture_pct']:.0f}% of all delays.")
    st.write("Uses only schedule information (airline, airports, date, time, distance) and historical delay rates from 2015 US DOT data. "
             "It does not see live weather or air-traffic conditions - treat it as a risk indicator, not a guarantee.")
