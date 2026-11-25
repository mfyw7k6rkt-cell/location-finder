"""Studio Intelligenza Artigianale – long-term location finder (v1.8).

Scores EU mainland cities on:
  1. market reach   – share of EU GDP / manufacturing reachable within N delivery days
  2. labour cost    – hourly labour cost incl. employer contributions
  3. talent pool    – employment in high & medium-high tech manufacturing nearby
Run locally:  streamlit run app.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

DATA = Path(__file__).parent / "data"

st.set_page_config(page_title="Location finder – Studio Intelligenza Artigianale", layout="wide")
st.title("Where to build: long-term location finder")
MAP_STYLE = __import__("os").environ.get("MAP_STYLE", "carto-positron")


# ---------- data ----------
@st.cache_data
def load():
    cities = pd.read_csv(DATA / "cities.csv")
    cities = cities.merge(pd.read_csv(DATA / "labour_cost.csv"), on="country_code", how="left")
    cities = cities.merge(pd.read_csv(DATA / "productivity.csv"), on="country_code", how="left")
    regions = pd.read_csv(DATA / "manufacturing_nuts3.csv")
    talent = pd.read_csv(DATA / "talent_nuts3.csv")
    firms = pd.read_csv(DATA / "firms_labour_nuts2.csv")
    cities = cities.merge(firms[["nuts2", "mfg_labour_cost_teur"]], on="nuts2", how="left")
    cities = cities.merge(pd.read_csv(DATA / "nuts2_names.csv"), on="nuts2", how="left")
    return cities, regions, talent, firms


def haversine_matrix(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(x, dtype=float)) for x in (lat1, lon1, lat2, lon2))
    dlat = lat2[None, :] - lat1[:, None]
    dlon = lon2[None, :] - lon1[:, None]
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1)[:, None] * np.cos(lat2)[None, :] * np.sin(dlon / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def delivery_days(road_km, km_per_day):
    return np.maximum(1, np.ceil(road_km / km_per_day)).astype(int)


cities, regions, talent, firms = load()

# 'Blue banana': Europe's historic economic corridor (approximate centre line, for orientation only)
_BB = [("Manchester", 53.48, -2.24), ("London", 51.51, -0.13), ("Brussels", 50.85, 4.35),
       ("Cologne", 50.94, 6.96), ("Frankfurt", 50.11, 8.68), ("Stuttgart", 48.78, 9.18),
       ("Zurich", 47.38, 8.54), ("Milan", 45.46, 9.19), ("Genoa", 44.41, 8.93)]


# Major mainland-EU seaports (selection; approximate port locations)
PORTS = pd.DataFrame([
    ("Rotterdam", 51.95, 4.14), ("Antwerp-Bruges", 51.27, 4.33), ("Zeebrugge", 51.33, 3.2), ("Hamburg", 53.54, 9.97),
    ("Bremerhaven", 53.56, 8.56), ("Wilhelmshaven", 53.6, 8.15), ("Le Havre", 49.48, 0.11), ("Dunkirk", 51.05, 2.37),
    ("Marseille-Fos", 43.4, 4.9), ("Genoa", 44.41, 8.92), ("La Spezia", 44.1, 9.84), ("Livorno", 43.55, 10.3),
    ("Trieste", 45.65, 13.77), ("Venice", 45.44, 12.26), ("Ravenna", 44.49, 12.28), ("Gioia Tauro", 38.44, 15.9),
    ("Koper", 45.55, 13.73), ("Rijeka", 45.33, 14.42), ("Barcelona", 41.35, 2.16), ("Valencia", 39.44, -0.32),
    ("Algeciras", 36.13, -5.43), ("Bilbao", 43.35, -3.05), ("Sines", 37.95, -8.87), ("Lisbon", 38.7, -9.15),
    ("Leixões", 41.18, -8.7), ("Gdańsk", 54.4, 18.68), ("Gdynia", 54.53, 18.55), ("Szczecin", 53.43, 14.56),
    ("Aarhus", 56.15, 10.22), ("Gothenburg", 57.69, 11.9), ("Piraeus", 37.94, 23.6), ("Thessaloniki", 40.63, 22.93),
    ("Constanța", 44.17, 28.65)], columns=["port", "lat", "lon"])


def banana_trace():
    return go.Scattermap(lat=[p[1] for p in _BB], lon=[p[2] for p in _BB], mode="lines",
                         line={"width": 14, "color": "rgba(30, 90, 200, 0.25)"},
                         name="Blue banana", hoverinfo="name")


# ---------- sidebar ----------
st.sidebar.header("Scoring weights")
WEIGHTS = {  # key: (label, default %, help)
    "w_reach": ("Market reach", 36, "Share of EU demand within the delivery promise."),
    "w_labour": ("Labour cost", 36, "Lower cost scores higher."),
    "w_talent": ("Talent pool", 18, "Employment in high & medium-high tech manufacturing within the hiring radius."),
    "w_port": ("Port proximity", 10, "Straight-line km to the nearest major EU seaport; closer scores higher."),
}
for _k, (_l, _d, _h) in WEIGHTS.items():
    st.session_state.setdefault(_k, _d)


def rebalance(changed):
    """Keep the weights at 100% in total: the other sliders absorb the change in proportion to their size."""
    others = [k for k in WEIGHTS if k != changed]
    rest = 100 - st.session_state[changed]
    total = sum(st.session_state[k] for k in others)
    if total == 0:
        new = {k: rest / len(others) for k in others}
    else:
        new = {k: st.session_state[k] * rest / total for k in others}
    rounded = {k: int(round(v)) for k, v in new.items()}
    diff = rest - sum(rounded.values())
    if diff:
        top_k = max(rounded, key=lambda k: new[k])
        rounded[top_k] += diff
    for k, v in rounded.items():
        st.session_state[k] = max(0, min(100, v))


for _k, (_l, _d, _h) in WEIGHTS.items():
    st.sidebar.slider(f"{_l} (%)", 0, 100, key=_k, help=_h, on_change=rebalance, args=(_k,))
w_reach, w_labour, w_talent, w_port = (st.session_state[k] for k in WEIGHTS)
w_rent = 0  # industrial rent removed: no reliable rent data across EU cities
w_sum = max(w_reach + w_labour + w_talent + w_rent + w_port, 1)
st.sidebar.caption(f"Total: {w_sum}%. Moving one slider rescales the others so the total stays 100%.")

st.sidebar.header("Labour & talent")
labour_source = st.sidebar.radio(
    "Labour cost basis", ["Regional, manufacturing (k€/person/yr)", "National, all sectors (€/hour)"], index=0,
    help="Regional: Eurostat SBS labour cost per person employed in manufacturing, NUTS-2, 2024. "
         "National: Eurostat hourly labour cost, whole economy, 2025.")
hire_km = st.sidebar.slider("Hiring radius (km)", 50, 300, 150, 25, help="Talent is distance-weighted: full weight nearby, 37% at this radius, fading to zero beyond about twice the radius.")

st.sidebar.header("Delivery promise")
promise_days = st.sidebar.slider("Delivery days", 1, 4, 2)
km_per_day = st.sidebar.slider("Km per delivery day (straight line)", 200, 1000, 500, 50)
road_factor = 1.0  # distances are straight-line ("as the crow flies")
show_banana = st.sidebar.checkbox("Show 'blue banana' corridor on map", value=True)

st.sidebar.header("Candidate cities")
# fixed search area: 1,050 km around 48.5°N, 8.0°E (northern Spain to Poland, northern Italy to Denmark)
c_lat, c_lon, radius = 48.5, 8.0, 1050
min_pop = st.sidebar.select_slider(
    "Min. city population", options=[50_000, 100_000, 200_000, 500_000], value=50_000,
    format_func=lambda v: f"{v:,}",
)

# ---------- candidates ----------
dist_centre = haversine_matrix([c_lat], [c_lon], cities.lat, cities.lon)[0]
mask = (
    (dist_centre <= radius)
    & (cities.population >= min_pop)
)
cand = cities[mask].copy().reset_index(drop=True)
if cand.empty:
    st.warning("No candidate cities with these settings.")
    st.stop()

# demand points: manufacturing firms (local units) per NUTS-2 region
dem = firms.dropna(subset=["mfg_firms"]).rename(columns={"mfg_firms": "weight", "nuts2": "name"})[
    ["name", "lat", "lon", "weight"]]
dem_label = "EU manufacturing firms"
w = dem.weight.to_numpy(float)
w_total = w.sum()

# 1. market reach
R_all = haversine_matrix(cand.lat, cand.lon, dem.lat, dem.lon) * road_factor  # candidates x demand, road km
days = delivery_days(R_all, km_per_day)
cand["reach_pct"] = 100 * (w[None, :] * (days <= promise_days)).sum(axis=1) / w_total
for _k in (1, 2, 3):
    cand[f"reach_{_k}d_pct"] = 100 * (w[None, :] * (days <= _k)).sum(axis=1) / w_total

# 2. labour cost
if labour_source.startswith("Regional"):
    cand["labour_metric"] = cand.mfg_labour_cost_teur
    lab_label = "mfg labour k€/yr"
else:
    cand["labour_metric"] = cand.labour_cost_eur_h
    lab_label = "labour €/h"

cand["rent"] = 0.0  # rent not used (no reliable data)

# 3. talent pool (thousand people in high/medium-high tech manufacturing within hiring radius)
Dt = haversine_matrix(cand.lat, cand.lon, talent.lat, talent.lon)
# distance-weighted (no hard cut-off): weight 1 at 0 km, 0.37 at the hiring radius, ~0 beyond twice the radius
cand["talent_ths"] = (talent.hitech_mfg_ths.to_numpy(float)[None, :] * np.exp(-(Dt / hire_km) ** 2)).sum(axis=1)

# 5. port proximity (road km to nearest major seaport)
Dp = haversine_matrix(cand.lat, cand.lon, PORTS.lat, PORTS.lon) * road_factor
cand["port_km"] = Dp.min(axis=1)
cand["nearest_port"] = PORTS.port.to_numpy()[Dp.argmin(axis=1)]

# score scaling bounds from single-site candidates (reused for networks)
B = {c: (cand[c].min(), cand[c].max()) for c in ("reach_pct", "labour_metric", "talent_ths", "rent", "port_km")}


def scale(x, col, higher_is_better=True):
    lo, hi = B[col]
    s = 100 * (np.asarray(x, float) - lo) / max(hi - lo, 1e-9)
    return s if higher_is_better else 100 - s


def score_of(reach, labour, tal, rnt, prt):
    return (w_reach * scale(reach, "reach_pct") + w_labour * scale(labour, "labour_metric", False)
            + w_talent * scale(tal, "talent_ths") + w_rent * scale(rnt, "rent", False)
            + w_port * scale(prt, "port_km", False)) / w_sum


cand["score"] = score_of(cand.reach_pct, cand.labour_metric, cand.talent_ths, cand.rent, cand.port_km)
order_idx = cand.sort_values(["score", "reach_pct"], ascending=False).index.to_numpy()
cand = cand.loc[order_idx].reset_index(drop=True)
R_all = R_all[order_idx]
cand.insert(0, "rank", cand.index + 1)
LAB = cand.labour_metric.to_numpy(float)
TAL = cand.talent_ths.to_numpy(float)
RENT = cand.rent.to_numpy(float)
PORT = cand.port_km.to_numpy(float)
labels = (cand.city + " (" + cand.country_code + ")").tolist()


def network_metrics(site_idx):
    """Each demand point is served by its nearest site; labour & talent weighted by demand served."""
    Rs = R_all[site_idx]
    nearest = Rs.argmin(axis=0)
    dmin = Rs.min(axis=0)
    dd = delivery_days(dmin, km_per_day)
    lab = (w * LAB[site_idx][nearest]).sum() / w_total
    tal = (w * TAL[site_idx][nearest]).sum() / w_total
    rnt = (w * RENT[site_idx][nearest]).sum() / w_total
    prt = (w * PORT[site_idx][nearest]).sum() / w_total
    out = {f"reach {k}d %": 100 * (w * (dd <= k)).sum() / w_total for k in (1, 2, 3)}
    reach_p = 100 * (w * (dd <= promise_days)).sum() / w_total
    out.update({"avg " + lab_label: lab, "talent (k)": tal, "rent €/m²": rnt, "avg port km": prt,
                "score": float(score_of(reach_p, lab, tal, rnt, prt))})
    served = np.array([(w * (nearest == i)).sum() for i in range(len(site_idx))]) / w_total * 100
    return out, nearest, dmin, served


def greedy_network(k):
    """Greedy: add the site that most improves the network score, up to k sites."""
    chosen = []
    dmin = np.full(len(dem), np.inf)
    labv, talv, rentv, portv = (np.zeros(len(dem)) for _ in range(4))

    def avg(m):
        return (w[None, :] * m).sum(axis=1) / w_total

    for _ in range(k):
        closer = R_all < dmin[None, :]
        new_d = np.where(closer, R_all, dmin[None, :])
        new_lab = np.where(closer, LAB[:, None], labv[None, :])
        new_tal = np.where(closer, TAL[:, None], talv[None, :])
        new_rent = np.where(closer, RENT[:, None], rentv[None, :])
        new_port = np.where(closer, PORT[:, None], portv[None, :])
        reach = 100 * (w[None, :] * (delivery_days(new_d, km_per_day) <= promise_days)).sum(axis=1) / w_total
        sc = score_of(reach, avg(new_lab), avg(new_tal), avg(new_rent), avg(new_port))
        sc[chosen] = -np.inf
        best = int(np.argmax(sc))
        chosen.append(best)
        dmin, labv, talv, rentv, portv = new_d[best], new_lab[best], new_tal[best], new_rent[best], new_port[best]
    return chosen


DAY_COLORS = {"1 day": "#1a9850", "2 days": "#a6d96a", "3 days": "#fdae61", "4+ days": "#d73027"}
DAY_ORDER = list(DAY_COLORS)


def day_label(d):
    return np.where(d >= 4, "4+ days", d.astype(str) + np.where(d > 1, " days", " day"))


# ---------- header ----------
st.caption(
    f"{len(cand)} EU mainland candidate cities. Market reach = share of {dem_label} within {promise_days} "
    f"delivery day(s) at 1 day per {km_per_day} km straight-line distance. Talent = people employed in high & medium-high tech "
    f"manufacturing within {hire_km} km."
)

top = cand.iloc[0]
k1, k2, k3, k4 = st.columns(4)
k1.metric("Top location", f"{top.city} ({top.country_code})")
k1.caption(f"Score {top.score:.0f} / 100")
k2.metric(f"Market reach ({promise_days}d)", f"{top.reach_pct:.0f}%")
k3.metric("Labour cost (mfg, regional)", f"€{top.mfg_labour_cost_teur:.0f}k/yr")
k3.caption(f"National, all sectors: €{top.labour_cost_eur_h:.1f}/h")
k4.metric(f"Talent within {hire_km} km", f"{top.talent_ths:,.0f}k")

# ---------- map of candidates ----------
st.subheader("Map of candidate locations")
if True:
    fig = px.scatter_map(
        cand, lat="lat", lon="lon", color="score", size="population", size_max=22, hover_name="city",
        hover_data={"country": True, "rank": True, "score": ":.0f", "reach_pct": ":.0f",
                    "labour_cost_eur_h": ":.1f", "mfg_labour_cost_teur": ":.0f", "talent_ths": ":,.0f",
                    "nearest_port": True, "port_km": ":,.0f", "lat": False, "lon": False, "population": ":,"},
        color_continuous_scale="Viridis", zoom=4, center={"lat": c_lat, "lon": c_lon},
        map_style=MAP_STYLE, height=600,
    )
    ang = np.linspace(0, 2 * np.pi, 120)
    fig.add_trace(go.Scattermap(
        lat=c_lat + (radius / 111.0) * np.sin(ang),
        lon=c_lon + (radius / (111.0 * np.cos(np.radians(c_lat)))) * np.cos(ang),
        mode="lines", line={"width": 1, "color": "grey"}, hoverinfo="skip", showlegend=False))
    if show_banana:
        fig.add_trace(banana_trace())
    fig.add_trace(go.Scattermap(lat=PORTS.lat, lon=PORTS.lon, mode="markers", text=PORTS.port, hoverinfo="text",
                                marker={"size": 9, "color": "#990f3d"}, name="Major ports"))
    fig.update_layout(margin={"l": 0, "r": 0, "t": 0, "b": 0})
    st.plotly_chart(fig, width="stretch")

TABLE_COLS = {"reach_1d_pct": "reach 1d %", "reach_2d_pct": "reach 2d %", "reach_3d_pct": "reach 3d %",
              "mfg_labour_cost_teur": "mfg k€/yr", "labour_cost_eur_h": "€/h", "talent_ths": "talent (k)",
              "rent": "rent €/m²", "port_km": "port km", "nearest_port": "nearest port"}
if w_rent == 0:  # hide rent until it is used in the score
    TABLE_COLS.pop("rent")
NUM_FMT = {"reach 1d %": "%.0f", "reach 2d %": "%.0f", "reach 3d %": "%.0f", "€/h": "%.1f", "mfg k€/yr": "%.0f",
           "rent €/m²": "%.2f", "port km": "%.0f",
           "talent (k)": "%,.0f", "score": "%.0f"}

st.subheader("Top 20 locations")
one_per_region = st.toggle("Show only the best city per region", value=True,
                           help="Avoids a list of neighbouring cities from the same region.")
top_src = cand.drop_duplicates("nuts2") if one_per_region else cand
show = top_src.head(20)[["rank", "city", "region", "country_code", "score", *TABLE_COLS]].rename(
    columns={"country_code": "cc", **TABLE_COLS})
cfg = {c: st.column_config.NumberColumn(format=f) for c, f in NUM_FMT.items() if c in show}
cfg["score"] = st.column_config.ProgressColumn("score", min_value=0, max_value=100, format="%.0f")
st.dataframe(show, hide_index=True, width="stretch", column_config=cfg, height=36 * 21 + 3)

# ---------- regions ----------
st.subheader("Top regions")
st.caption("Regions (NUTS-2) ranked by their best-scoring city. 'Cities in top 50' shows how deep the region is.")
top50 = set(cand.head(50).index)
reg = (cand.assign(in_top50=cand.index.isin(top50))
       .groupby(["nuts2", "region", "country_code"], as_index=False)
       .agg(best_score=("score", "max"), best_city=("city", "first"), best_rank=("rank", "min"),
            avg_score=("score", "mean"), cities=("city", "count"), in_top50=("in_top50", "sum"),
            reach=("reach_pct", "max"), labour=("mfg_labour_cost_teur", "mean"), talent=("talent_ths", "max"))
       .sort_values("best_score", ascending=False).reset_index(drop=True))
reg.insert(0, "region rank", reg.index + 1)
st.dataframe(
    reg.head(20)[["region rank", "region", "country_code", "best_city", "best_rank", "best_score", "avg_score",
                  "cities", "in_top50", "reach", "labour", "talent"]].rename(columns={
        "country_code": "cc", "best_city": "best city", "best_rank": "city rank", "best_score": "best score",
        "avg_score": "avg score", "cities": "candidate cities", "in_top50": "cities in top 50",
        "reach": f"reach {promise_days}d %", "labour": "mfg k€/yr", "talent": "talent (k)"}),
    hide_index=True, width="stretch",
    column_config={"best score": st.column_config.ProgressColumn("best score", min_value=0, max_value=100, format="%.0f"),
                   "avg score": st.column_config.NumberColumn(format="%.0f"),
                   f"reach {promise_days}d %": st.column_config.NumberColumn(format="%.0f"),
                   "mfg k€/yr": st.column_config.NumberColumn(format="%.0f"),
                   "talent (k)": st.column_config.NumberColumn(format="%,.0f")})

# ---------- compare cities ----------
st.subheader("Compare cities")
defaults = [l for l in ["Berlin (DE)", "La Spezia (IT)", "Munich (DE)", "Stuttgart (DE)", "Eindhoven (NL)",
                        "Milan (IT)", "Prague (CZ)", labels[0]] if l in labels]
chosen = st.multiselect("Cities", labels, default=list(dict.fromkeys(defaults)))
if chosen:
    comp = cand.set_index(pd.Index(labels)).loc[chosen]
    comp = comp[["rank", "score", *TABLE_COLS]].rename(columns=TABLE_COLS)
    st.dataframe(comp, width="stretch",
                 column_config={c: st.column_config.NumberColumn(format=f) for c, f in NUM_FMT.items() if c in comp})

# ---------- multi-site network ----------
st.subheader("Production network: 1 to 5 sites")
st.caption("Each customer region is served from its nearest site; labour cost, talent and port distance are weighted "
           "by the demand each site serves. Gross score uses the single-site scale (above 100 = better than any single "
           "site). Net score subtracts a cost for every site beyond the first, so more sites only win if they add enough.")
site_cost = st.slider("Cost per extra site (score points)", 0, 20, 8,
                      help="How many score points an extra site must add to be worth its fixed costs "
                           "(rent, equipment, staff, management). 0 = sites are free, so more sites always win.")

chain = greedy_network(5)  # greedy: each step adds the site that improves the network most
opt_rows = []
for k in range(1, 6):
    m, *_ = network_metrics(chain[:k])
    net = m["score"] - site_cost * (k - 1)
    opt_rows.append({"sites": k, "locations": " + ".join(cand.city[i] for i in chain[:k]),
                     **{c: v for c, v in m.items() if c != "rent €/m²"}, "site cost": -site_cost * (k - 1),
                     "net score": net})
opt = pd.DataFrame(opt_rows).rename(columns={"score": "gross score"})
best_k = int(opt.loc[opt["net score"].idxmax(), "sites"])
opt.insert(0, "", np.where(opt.sites == best_k, "recommended", ""))
st.markdown(f"**Suggested networks** (greedy optimisation on current settings). Recommended: **{best_k} "
            f"site{'s' if best_k > 1 else ''}**.")
st.dataframe(opt, hide_index=True, width="stretch",
             column_config={c: st.column_config.NumberColumn(format="%.0f")
                            for c in opt.columns if c not in ("", "sites", "locations")})

mode = st.radio("Network to inspect", ["Recommended", "1 site", "2 sites", "3 sites", "4 sites", "5 sites",
                                       "Choose manually"], horizontal=True)
if mode == "Choose manually":
    manual = st.multiselect("Select up to 5 sites", labels, max_selections=5,
                            default=[labels[i] for i in chain[:best_k]])
    net_idx = [labels.index(l) for l in manual]
else:
    net_idx = chain[:best_k] if mode == "Recommended" else chain[:int(mode[0])]

if net_idx:
    m, nearest, dmin, served = network_metrics(net_idx)
    a, b, c, d = st.columns(4)
    a.metric("Within 1 day", f"{m['reach 1d %']:.0f}%")
    b.metric("Within 2 days", f"{m['reach 2d %']:.0f}%")
    c.metric("Within 3 days", f"{m['reach 3d %']:.0f}%")
    d.metric("Net score", f"{m['score'] - site_cost * (len(net_idx) - 1):.0f}")
    d.caption(f"Gross {m['score']:.0f} − site cost {site_cost * (len(net_idx) - 1)}")

    nd = dem.copy()
    nd["dist_km"] = dmin
    nd["delivery"] = day_label(delivery_days(dmin, km_per_day))
    nd["served by"] = [cand.city[net_idx[i]] for i in nearest]
    n1, n2 = st.columns([3, 2])
    with n1:
        fig3 = px.scatter_map(
            nd, lat="lat", lon="lon", color="delivery", size="weight", size_max=16, hover_name="name",
            hover_data={"served by": True, "dist_km": ":,.0f", "lat": False, "lon": False, "weight": ":,.0f"},
            color_discrete_map=DAY_COLORS, category_orders={"delivery": DAY_ORDER},
            zoom=3.3, center={"lat": c_lat, "lon": c_lon}, map_style=MAP_STYLE, height=520,
        )
        sites = cand.iloc[net_idx]
        fig3.add_trace(go.Scattermap(lat=sites.lat, lon=sites.lon, mode="markers+text", text=sites.city,
                                     textposition="top right", marker={"size": 18, "color": "black"},
                                     name="Sites", hoverinfo="text"))
        fig3.update_layout(margin={"l": 0, "r": 0, "t": 0, "b": 0})
        st.plotly_chart(fig3, width="stretch")
    with n2:
        st.markdown("**Per site**")
        st.dataframe(pd.DataFrame({"site": sites.city.values, "cc": sites.country_code.values,
                                   "mfg k€/yr": sites.mfg_labour_cost_teur.values,
                                   "€/h": sites.labour_cost_eur_h.values,
                                   **({"rent €/m²": sites.rent.values} if w_rent else {}),
                                   "nearest port": (sites.nearest_port + " (" + sites.port_km.round().astype(int).astype(str) + " km)").values,
                                   "talent (k)": sites.talent_ths.values, "demand served %": served}),
                     hide_index=True, width="stretch",
                     column_config={"demand served %": st.column_config.NumberColumn(format="%.0f"),
                                    "€/h": st.column_config.NumberColumn(format="%.1f"),
                                    "mfg k€/yr": st.column_config.NumberColumn(format="%.0f"),
                                    "rent €/m²": st.column_config.NumberColumn(format="%.2f"),
                                    "talent (k)": st.column_config.NumberColumn(format="%,.0f")})

# ---------- city detail ----------
st.subheader("Delivery days from a selected city")
pick = st.selectbox("City", labels, index=0)
row = cand.iloc[labels.index(pick)]
dest = dem.copy()
dest["dist_km"] = haversine_matrix([row.lat], [row.lon], dem.lat, dem.lon)[0] * road_factor
dest["delivery"] = day_label(delivery_days(dest.dist_km, km_per_day))
c1, c2 = st.columns([3, 2])
with c1:
    fig2 = px.scatter_map(
        dest, lat="lat", lon="lon", color="delivery", size="weight", size_max=16, hover_name="name",
        hover_data={"dist_km": ":,.0f", "lat": False, "lon": False, "weight": ":,.0f"},
        color_discrete_map=DAY_COLORS, category_orders={"delivery": DAY_ORDER},
        zoom=3.3, center={"lat": row.lat, "lon": row.lon}, map_style=MAP_STYLE, height=520,
    )
    fig2.add_trace(go.Scattermap(lat=[row.lat], lon=[row.lon], mode="markers", marker={"size": 16, "color": "black"},
                                 name=row.city, hoverinfo="name"))
    fig2.update_layout(margin={"l": 0, "r": 0, "t": 0, "b": 0})
    st.plotly_chart(fig2, width="stretch")
with c2:
    share = dest.groupby("delivery").weight.sum().reindex(DAY_ORDER).fillna(0)
    cum = (100 * share.cumsum() / share.sum()).round(0)
    st.markdown(f"**Cumulative share of {dem_label} reached**")
    for lbl, v in zip(["Within 1 day", "Within 2 days", "Within 3 days"], cum.iloc[:3]):
        st.metric(lbl, f"{v:.0f}%")
    st.metric("Labour cost (manufacturing, regional)", f"€{row.mfg_labour_cost_teur:.0f}k per person/yr")
    st.caption(f"National, all sectors: €{row.labour_cost_eur_h:.1f}/h")
    st.metric(f"Talent within {hire_km} km", f"{row.talent_ths:,.0f}k")

# ---------- sources ----------
with st.expander("Data, sources and limitations"):
    st.markdown(
        """
1. **Cities**: GeoNames (via `geonamescache`), EU-27 mainland cities > 50,000 inhabitants; districts within 15 km merged. Ireland, Cyprus, Malta and islands excluded.
2. **Demand**: number of manufacturing local units (firms and plants, micro firms included) per NUTS-2 region, Eurostat `sbs_r_nuts2021`, 2024, at the GDP-weighted centre of each region. Mainland EU only.
3. **Delivery days**: distance model, not carrier data. Straight-line distance; 1 day per set number of km.
4. **Labour cost** (choose in sidebar):
   - Regional (default): Eurostat `sbs_r_nuts2021`, labour cost per person employed in manufacturing local units, NUTS-2, 2024, incl. employer contributions. "Persons employed" also counts unpaid owners and family workers, which lowers the average in regions with many micro firms (e.g. Italy).
   - National: Eurostat `lc_lci_lev`, hourly labour costs, whole economy excl. public administration, 2025 (Belgium 2024).
   **Firms**: number of manufacturing local units per NUTS-2 region, same dataset, 2024 (includes micro firms).
5. **Port proximity**: straight-line km to the nearest of 33 major mainland-EU seaports (a selection, approximate locations); shorter scores higher.
6. **Talent**: employment in high & medium-high technology manufacturing (Eurostat `htec_emp_reg2`, NACE C_HTC_MH, thousand persons, NUTS-2, latest year), spread over its NUTS-3 regions in proportion to manufacturing GVA (so talent sits where industry is), then distance-weighted from each city (weight 1 nearby, 37% at the hiring radius, fading beyond).
7. The 'blue banana' line is an approximate centre line for orientation only; it does not enter the score.
8. Not yet included: industrial rent, carrier cut-off times, actual shipping rates, R&D spending.
"""
    )
