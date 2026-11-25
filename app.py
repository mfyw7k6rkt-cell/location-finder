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
    cities = cities.merge(pd.read_csv(DATA / "regions_large.csv"), on="nuts2", how="left")
    return cities, regions, talent, firms


def haversine_matrix(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(x, dtype=float)) for x in (lat1, lon1, lat2, lon2))
    dlat = lat2[None, :] - lat1[:, None]
    dlon = lon2[None, :] - lon1[:, None]
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1)[:, None] * np.cos(lat2)[None, :] * np.sin(dlon / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


# Land mask (0.1° grid, lon -12..32, lat 34..62) from Natural Earth 1:110m land via sane-topojson europe_110m
LAND = np.unpackbits(np.load(DATA / "landmask.npy"))[:280 * 440].reshape(280, 440).astype(bool)
SEA_PENALTY = 1.5  # a route that is 100% over sea counts as 2.5x its straight-line length (detour or ferry)


@st.cache_data(show_spinner=False)
def eff_dist(lat1, lon1, lat2, lon2, n=16, edge_km=30.0):
    """Straight-line km, lengthened where the straight line crosses sea: km × (1 + 1.5 × share of the line over sea).
    The first and last 30 km are ignored so coastal cities are not penalised for their own coastline."""
    lat1, lon1, lat2, lon2 = (np.asarray(x, dtype=np.float32) for x in (lat1, lon1, lat2, lon2))
    D = haversine_matrix(lat1, lon1, lat2, lon2).astype(np.float32)
    t = np.linspace(0, 1, n, dtype=np.float32)
    edge = np.minimum(t, 1 - t)
    out = np.empty_like(D)
    for a in range(0, len(lat1), 64):  # chunk rows to keep memory low
        b = slice(a, a + 64)
        la = lat1[b, None, None] + (lat2[None, :, None] - lat1[b, None, None]) * t
        lo = lon1[b, None, None] + (lon2[None, :, None] - lon1[b, None, None]) * t
        i = np.clip(((la - 34) / 0.1).astype(np.int32), 0, 279)
        j = np.clip(((lo + 12) / 0.1).astype(np.int32), 0, 439)
        valid = edge[None, None, :] * D[b, :, None] > edge_km
        sea = (~LAND[i, j]) & valid
        frac = sea.sum(2) / np.maximum(valid.sum(2), 1)
        out[b] = D[b] * (1 + SEA_PENALTY * frac)
    return out


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
st.sidebar.header("Delivery promise")
promise_days = st.sidebar.slider("Delivery days", 1, 4, 2)
km_per_day = st.sidebar.slider("Km per delivery day (straight line)", 200, 1000, 500, 50)
road_factor = 1.0  # distances are straight-line ("as the crow flies")
show_banana = st.sidebar.checkbox("Show 'blue banana' corridor on map", value=True)

st.sidebar.header("Production sites")
n_sites_choice = st.sidebar.select_slider(
    "Number of production sites", options=["Auto", "1", "2", "3"], value="1",
    help="Auto = the number with the best net score (see 'Cost per extra site'). The chosen network is shown in the "
         "'Production network' section and on its map.")
site_cost = st.sidebar.slider("Cost per extra site (score points)", 0, 20, 8,
                              help="How many score points an extra site must add to be worth its fixed costs "
                                   "(rent, equipment, staff, management). 0 = sites are free, so more sites always win.")
region_level = "Large regions"  # German states, Italian macro-regions, Bohemia / Moravia-Silesia, ...

st.sidebar.header("Scoring weights")
WEIGHTS = {  # key: (label, default %, help)
    "w_reach": ("Market reach", 50, "Share of EU demand within the delivery promise."),
    "w_labour": ("Labour cost", 40, "Lower cost scores higher."),
    "w_talent": ("Talent pool", 5, "Employment in high & medium-high tech manufacturing within the hiring radius."),
    "w_port": ("Port proximity", 5, "Straight-line km to the nearest major EU seaport; closer scores higher."),
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
min_talent = st.sidebar.slider("Minimum talent pool (k people)", 0, 600, 250, 25,
                               help="Cities with fewer high-tech manufacturing workers within the hiring radius are "
                                    "excluded: a production site must be able to hire technicians and engineers.")
same_country = st.sidebar.toggle("Only count talent in the same country", value=True,
                                 help="Cross-border commuting is rare (e.g. few Bavarians work in Czechia), so by "
                                      "default only workers in the city's own country count.")
hire_km = st.sidebar.slider("Hiring radius (km)", 50, 300, 150, 25, help="Talent is distance-weighted: full weight nearby, 37% at this radius, fading to zero beyond about twice the radius.")

st.sidebar.header("Candidate cities")
# fixed search area: 1,050 km around 48.5°N, 8.0°E (northern Spain to Poland, northern Italy to Denmark)
c_lat, c_lon, radius = 48.5, 8.0, 1050
min_pop = st.sidebar.select_slider(
    "Min. city population", options=[50_000, 75_000, 100_000, 200_000, 500_000], value=75_000,
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
R_all = eff_dist(cand.lat.values, cand.lon.values, dem.lat.values, dem.lon.values)  # candidates x demand, km (sea-corrected)
days = delivery_days(R_all, km_per_day)
cand["reach_pct"] = 100 * (w[None, :] * (days <= promise_days)).sum(axis=1) / w_total
for _k in (1, 2, 3):
    cand[f"reach_{_k}d_pct"] = 100 * (w[None, :] * (days <= _k)).sum(axis=1) / w_total

# 2. labour cost: regional manufacturing labour cost per person per year
cand["labour_metric"] = cand.mfg_labour_cost_teur
lab_label = "mfg labour k€/yr"

cand["rent"] = 0.0  # rent not used (no reliable data)

# 3. talent pool (thousand people in high/medium-high tech manufacturing within hiring radius)
Dt = eff_dist(cand.lat.values, cand.lon.values, talent.lat.values, talent.lon.values)
# distance-weighted (no hard cut-off): weight 1 at 0 km, 0.37 at the hiring radius, ~0 beyond twice the radius
tw = talent.hitech_mfg_ths.to_numpy(float)[None, :] * np.exp(-(Dt / hire_km) ** 2)
if same_country:  # no cross-border commuting: only talent in the candidate's own country counts
    t_cc = talent.nuts3.str[:2].replace({"EL": "GR"}).to_numpy()
    tw = tw * (cand.country_code.to_numpy()[:, None] == t_cc[None, :])
cand["talent_ths"] = tw.sum(axis=1)

# 5. port proximity (road km to nearest major seaport)
Dp = eff_dist(cand.lat.values, cand.lon.values, PORTS.lat.values, PORTS.lon.values)
cand["port_km"] = Dp.min(axis=1)
cand["nearest_port"] = PORTS.port.to_numpy()[Dp.argmin(axis=1)]

# keep every city in the search area for manual network selection (no minimum talent there)
cand_full, R_full = cand.copy(), R_all.copy()

# minimum talent pool: a location must be able to hire technicians and engineers
keep = (cand.talent_ths >= min_talent).to_numpy()
if not keep.any():
    st.warning("No candidate city meets the minimum talent pool. Lower the threshold in the sidebar.")
    st.stop()
cand = cand[keep].reset_index(drop=True)
R_all = R_all[keep]

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

# full set (all cities in the search area), same scoring scale, for manual selection
cand_full["score"] = score_of(cand_full.reach_pct, cand_full.labour_metric, cand_full.talent_ths,
                              cand_full.rent, cand_full.port_km)
ord_full = cand_full.sort_values("score", ascending=False).index.to_numpy()
cand_full = cand_full.loc[ord_full].reset_index(drop=True)
R_full = R_full[ord_full]
labels_full = (cand_full.city + " (" + cand_full.country_code + ")").tolist()
SRC = {"main": (cand, R_all), "full": (cand_full, R_full)}


def network_metrics(site_idx, src="main"):
    """Each demand point is served by its nearest site; labour & talent weighted by demand served."""
    frame, Rsrc = SRC[src]
    Rs = Rsrc[site_idx]
    nearest = Rs.argmin(axis=0)
    dmin = Rs.min(axis=0)
    dd = delivery_days(dmin, km_per_day)
    pick = lambda col: frame[col].to_numpy(float)[site_idx][nearest]
    lab = (w * pick("labour_metric")).sum() / w_total
    tal = (w * pick("talent_ths")).sum() / w_total
    rnt = (w * pick("rent")).sum() / w_total
    prt = (w * pick("port_km")).sum() / w_total
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


# ---------- production network (computed before the header so the header can show it) ----------
chain = greedy_network(3)  # greedy: each step adds the site that improves the network most
opt_rows = []
for k in range(1, 4):
    m, *_ = network_metrics(chain[:k])
    net = m["score"] - site_cost * (k - 1)
    opt_rows.append({"sites": k, "locations": " + ".join(cand.city[i] for i in chain[:k]),
                     **{c: v for c, v in m.items() if c != "rent €/m²"}, "site cost": -site_cost * (k - 1),
                     "net score": net})
opt = pd.DataFrame(opt_rows).rename(columns={"score": "gross score"})
best_k = int(opt.loc[opt["net score"].idxmax(), "sites"])
opt.insert(0, "", np.where(opt.sites == best_k, "recommended", ""))
k_show = best_k if n_sites_choice == "Auto" else int(n_sites_choice)

# ---------- header ----------
st.caption(
    f"{len(cand)} EU mainland candidate cities. Market reach = share of {dem_label} within {promise_days} "
    f"delivery day(s) at 1 day per {km_per_day} km straight-line distance. Talent = people employed in high & medium-high tech "
    f"manufacturing within {hire_km} km."
)

top = cand.iloc[0]

# ---------- map of candidates ----------
st.subheader("Map of candidate locations")
if True:
    fig = px.scatter_map(
        cand, lat="lat", lon="lon", color="score", size="population", size_max=22, hover_name="city",
        hover_data={"country": True, "rank": True, "score": ":.0f", "reach_pct": ":.0f",
                    "mfg_labour_cost_teur": ":.0f", "talent_ths": ":,.0f",
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
    fig.update_layout(margin={"l": 0, "r": 0, "t": 0, "b": 0})
    st.plotly_chart(fig, width="stretch")

TABLE_COLS = {"reach_1d_pct": "reach 1d %", "reach_2d_pct": "reach 2d %", "reach_3d_pct": "reach 3d %",
              "mfg_labour_cost_teur": "mfg k€/yr", "talent_ths": "talent (k)",
              "rent": "rent €/m²", "port_km": "port km"}
if w_rent == 0:  # hide rent until it is used in the score
    TABLE_COLS.pop("rent")
NUM_FMT = {"reach 1d %": "%.0f", "reach 2d %": "%.0f", "reach 3d %": "%.0f", "€/h": "%.1f", "mfg k€/yr": "%.0f",
           "rent €/m²": "%.2f", "port km": "%.0f",
           "talent (k)": "%,.0f", "score": "%.0f"}

REG_KEY, REG_NAME = "region_large_code", "region_large"

st.subheader("Top 20 locations")
one_per_region = st.toggle("Show only the best city per region", value=True,
                           help="Avoids a list of neighbouring cities from the same region.")
top_src = cand.drop_duplicates(REG_KEY) if one_per_region else cand
show = top_src.head(20)[["rank", "city", REG_NAME, "country_code", "score", *TABLE_COLS]].rename(
    columns={"country_code": "cc", REG_NAME: "region", **TABLE_COLS})
cfg = {c: st.column_config.NumberColumn(format=f) for c, f in NUM_FMT.items() if c in show}
cfg["score"] = st.column_config.ProgressColumn("score", min_value=0, max_value=100, format="%.0f")
st.dataframe(show, hide_index=True, width="stretch", column_config=cfg, height=36 * 21 + 3)

# ---------- regions ----------
st.subheader("Top regions")
st.caption("Regions (German states, Italian macro-regions, Bohemia / Moravia-Silesia, ...) ranked by their best-scoring city. 'Cities in top 50' shows how deep the region is.")
top50 = set(cand.head(50).index)
reg = (cand.assign(in_top50=cand.index.isin(top50))
       .groupby([REG_KEY, REG_NAME, "country_code"], as_index=False)
       .agg(best_score=("score", "max"), best_city=("city", "first"), best_rank=("rank", "min"),
            avg_score=("score", "mean"), cities=("city", "count"), in_top50=("in_top50", "sum"),
            reach=("reach_pct", "max"), labour=("mfg_labour_cost_teur", "mean"), talent=("talent_ths", "max"))
       .sort_values("best_score", ascending=False).reset_index(drop=True))
reg.insert(0, "region rank", reg.index + 1)
st.dataframe(
    reg.head(20)[["region rank", REG_NAME, "country_code", "best_city", "best_rank", "best_score", "avg_score",
                  "cities", "in_top50", "reach", "labour", "talent"]].rename(columns={
        REG_NAME: "region", "country_code": "cc", "best_city": "best city", "best_rank": "city rank", "best_score": "best score",
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
st.subheader("Production network: 1 to 3 sites")
st.caption("Each customer region is served from its nearest site; labour cost, talent and port distance are weighted "
           "by the demand each site serves. Gross score uses the single-site scale (above 100 = better than any single "
           "site). Net score subtracts a cost for every site beyond the first, so more sites only win if they add enough.")

st.markdown(f"**Suggested networks** (greedy optimisation on current settings). Recommended: **{best_k} "
            f"site{'s' if best_k > 1 else ''}**.")
st.dataframe(opt, hide_index=True, width="stretch",
             column_config={c: st.column_config.NumberColumn(format="%.0f")
                            for c in opt.columns if c not in ("", "sites", "locations")})

st.markdown(f"**Showing: {k_show} site{'s' if k_show > 1 else ''}** "
            f"({'recommended' if n_sites_choice == 'Auto' else 'set in the sidebar'}).")
manual_on = st.toggle("Choose sites manually instead", value=False)
if manual_on:
    st.caption("Any city in the search area can be picked here, also those below the minimum talent pool.")
    manual = st.multiselect("Select up to 3 sites", labels_full, max_selections=3,
                            default=[labels[i] for i in chain[:k_show]])
    net_idx = [labels_full.index(l) for l in manual]
    net_src, net_frame = "full", cand_full
else:
    net_idx = chain[:k_show]
    net_src, net_frame = "main", cand

if net_idx:
    m, nearest, dmin, served = network_metrics(net_idx, net_src)
    a, b, c, d = st.columns(4)
    a.metric("Within 1 day", f"{m['reach 1d %']:.0f}%")
    b.metric("Within 2 days", f"{m['reach 2d %']:.0f}%")
    c.metric("Within 3 days", f"{m['reach 3d %']:.0f}%")
    d.metric("Net score", f"{m['score'] - site_cost * (len(net_idx) - 1):.0f}")
    d.caption(f"Gross {m['score']:.0f} − site cost {site_cost * (len(net_idx) - 1)}")

    nd = dem.copy()
    nd["dist_km"] = dmin
    nd["delivery"] = day_label(delivery_days(dmin, km_per_day))
    nd["served by"] = [net_frame.city[net_idx[i]] for i in nearest]
    n1, n2 = st.columns([3, 2])
    with n1:
        fig3 = px.scatter_map(
            nd, lat="lat", lon="lon", color="delivery", size="weight", size_max=16, hover_name="name",
            hover_data={"served by": True, "dist_km": ":,.0f", "lat": False, "lon": False, "weight": ":,.0f"},
            color_discrete_map=DAY_COLORS, category_orders={"delivery": DAY_ORDER},
            zoom=3.3, center={"lat": c_lat, "lon": c_lon}, map_style=MAP_STYLE, height=520,
        )
        sites = net_frame.iloc[net_idx]
        fig3.add_trace(go.Scattermap(lat=sites.lat, lon=sites.lon, mode="markers+text", text=sites.city,
                                     textposition="top right", marker={"size": 18, "color": "black"},
                                     name="Sites", hoverinfo="text"))
        fig3.update_layout(margin={"l": 0, "r": 0, "t": 0, "b": 0})
        st.plotly_chart(fig3, width="stretch")
    with n2:
        st.markdown("**Per site**")
        st.dataframe(pd.DataFrame({"site": sites.city.values, "cc": sites.country_code.values,
                                   "mfg k€/yr": sites.mfg_labour_cost_teur.values,
                                   **({"rent €/m²": sites.rent.values} if w_rent else {}),
                                   "port km": sites.port_km.values,
                                   "talent (k)": sites.talent_ths.values, "demand served %": served}),
                     hide_index=True, width="stretch",
                     column_config={"demand served %": st.column_config.NumberColumn(format="%.0f"),
                                    "mfg k€/yr": st.column_config.NumberColumn(format="%.0f"),
                                    "port km": st.column_config.NumberColumn(format="%.0f"),
                                    "rent €/m²": st.column_config.NumberColumn(format="%.2f"),
                                    "talent (k)": st.column_config.NumberColumn(format="%,.0f")})

# ---------- city detail ----------
st.subheader("Delivery days from a selected city")
pick = st.selectbox("City", labels, index=0)
row = cand.iloc[labels.index(pick)]
dest = dem.copy()
dest["dist_km"] = eff_dist(np.array([row.lat]), np.array([row.lon]), dem.lat.values, dem.lon.values)[0]
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
    st.metric(f"Talent within {hire_km} km", f"{row.talent_ths:,.0f}k")

# ---------- sources ----------
with st.expander("Data, sources and limitations"):
    st.markdown(
        """
1. **Cities**: GeoNames (via `geonamescache`), EU-27 mainland cities > 50,000 inhabitants; districts within 15 km merged. Ireland, Cyprus, Malta and islands excluded.
2. **Demand**: number of manufacturing local units (firms and plants, micro firms included) per NUTS-2 region, Eurostat `sbs_r_nuts2021`, 2024, at the GDP-weighted centre of each region. Mainland EU only.
3. **Delivery days**: distance model, not carrier data. Straight-line distance, lengthened where the line crosses sea (× (1 + 1.5 × share over sea), using a Natural Earth land mask), so e.g. Croatia–Italy is not treated as a short hop across the Adriatic. 1 day per set number of km. The same correction applies to talent and port distances.
4. **Labour cost**: Eurostat `sbs_r_nuts2021`, labour cost per person employed in manufacturing local units, NUTS-2, 2024, incl. employer contributions. "Persons employed" also counts unpaid owners and family workers, which lowers the average in regions with many micro firms (e.g. Italy).
   **Firms**: number of manufacturing local units per NUTS-2 region, same dataset, 2024 (includes micro firms).
5. **Port proximity**: straight-line km to the nearest of 33 major mainland-EU seaports (a selection, approximate locations); shorter scores higher.
6. **Talent**: employment in high & medium-high technology manufacturing (Eurostat `htec_emp_reg2`, NACE C_HTC_MH, thousand persons, NUTS-2, latest year), spread over its NUTS-3 regions in proportion to manufacturing GVA (so talent sits where industry is), then distance-weighted from each city (weight 1 nearby, 37% at the hiring radius, fading beyond). By default only talent in the city's own country counts (no cross-border commuting).
7. The 'blue banana' line is an approximate centre line for orientation only; it does not enter the score.
8. Not yet included: industrial rent, carrier cut-off times, actual shipping rates, R&D spending.
"""
    )
