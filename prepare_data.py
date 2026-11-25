"""Builds the static data files for the location app.

Run once:  python prepare_data.py
Outputs:   data/cities.csv  (all EU cities > 50k inhabitants, districts merged)
           data/labour_cost.csv  (Eurostat hourly labour cost per country)
"""
from math import radians, sin, cos, asin, sqrt
from pathlib import Path

import geonamescache
import numpy as np
import pandas as pd

EU = {
    "AT", "BE", "BG", "HR", "CY", "CZ", "DK", "EE", "FI", "FR", "DE", "GR", "HU", "IE",
    "IT", "LV", "LT", "LU", "MT", "NL", "PL", "PT", "RO", "SK", "SI", "ES", "SE",
}
MIN_POP = 50_000
MERGE_KM = 15  # city districts closer than this to a larger city are merged into it

COUNTRY_NAMES = {
    "AT": "Austria", "BE": "Belgium", "BG": "Bulgaria", "HR": "Croatia", "CY": "Cyprus",
    "CZ": "Czechia", "DK": "Denmark", "EE": "Estonia", "FI": "Finland", "FR": "France",
    "DE": "Germany", "GR": "Greece", "HU": "Hungary", "IE": "Ireland", "IT": "Italy",
    "LV": "Latvia", "LT": "Lithuania", "LU": "Luxembourg", "MT": "Malta", "NL": "Netherlands",
    "PL": "Poland", "PT": "Portugal", "RO": "Romania", "SK": "Slovakia", "SI": "Slovenia",
    "ES": "Spain", "SE": "Sweden",
}

# Eurostat lc_lci_lev, hourly labour costs in EUR, whole economy excl. public administration
# (NACE B-S excl. O), wages + non-wage costs, 2025. Belgium: 2025 not published; 2024 value used.
LABOUR_COST_EUR_H = {
    "BG": 12.0, "CZ": 19.8, "DK": 51.7, "DE": 45.0, "EE": 21.1, "IE": 44.2, "GR": 18.2,
    "ES": 26.4, "FR": 44.3, "HR": 18.4, "IT": 32.0, "CY": 21.7, "LV": 16.3, "LT": 17.8,
    "LU": 56.8, "HU": 15.2, "MT": 19.0, "NL": 47.9, "AT": 46.3, "PL": 19.1, "PT": 19.4,
    "RO": 13.6, "SI": 29.7, "SK": 19.8, "FI": 39.4, "SE": 43.1, "BE": 48.2,
}
LABOUR_YEAR = {c: 2025 for c in LABOUR_COST_EUR_H} | {"BE": 2024}


def km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(radians, (lat1, lon1, lat2, lon2))
    a = sin((lat2 - lat1) / 2) ** 2 + cos(lat1) * cos(lat2) * sin((lon2 - lon1) / 2) ** 2
    return 6371 * 2 * asin(sqrt(a))


GVA_URL = (
    "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/nama_10r_3gva"
    "?format=JSON&lang=EN&nace_r2=C&nace_r2=B-E&unit=CP_MEUR&sinceTimePeriod=2020"
)
GDP_URL = (
    "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/nama_10r_3gdp"
    "?format=JSON&lang=EN&unit=MIO_EUR&sinceTimePeriod=2021"
)
NUTS_PT_URL ="https://raw.githubusercontent.com/eurostat/Nuts2json/master/pub/v2/2024/4326/nutspt_3.json"
GVA_YEARS = ["2023", "2022", "2021"]  # most recent year with full NUTS-3 coverage first
EUROSTAT_TO_ISO = {"EL": "GR"}


# Not on the European mainland: island states, islands and overseas/exclave regions (NUTS prefixes)
OFFSHORE_COUNTRIES = {"IE", "CY", "MT"}
OFFSHORE_NUTS = (
    "ES53", "ES63", "ES64", "ES70",          # Balearics, Ceuta, Melilla, Canaries
    "PT2", "PT3",                             # Azores, Madeira
    "FRM", "FRY",                             # Corsica, overseas departments
    "ITG1", "ITG2",                           # Sicily, Sardinia
    "EL41", "EL42", "EL43", "EL62",           # Aegean islands, Crete, Ionian islands
    "FI2",                                    # Åland
    "SE214",                                  # Gotland
    "DK014",                                  # Bornholm
)


def is_offshore(nuts_id):
    cc = EUROSTAT_TO_ISO.get(nuts_id[:2], nuts_id[:2])
    return cc in OFFSHORE_COUNTRIES or nuts_id.startswith(OFFSHORE_NUTS)


def fetch_json(url):
    import json
    import urllib.request

    with urllib.request.urlopen(url, timeout=300) as r:
        return json.load(r)


def build_manufacturing(out):
    """NUTS-3 manufacturing GVA (Eurostat nama_10r_3gva) located at NUTS-3 label points."""
    d = fetch_json(GVA_URL)
    dims, size = d["id"], d["size"]
    cat = {k: d["dimension"][k]["category"]["index"] for k in dims}
    strides = [1] * len(size)
    for i in range(len(size) - 2, -1, -1):
        strides[i] = strides[i + 1] * size[i + 1]

    def get(**kw):
        return d["value"].get(str(sum(cat[k][kw[k]] * strides[i] for i, k in enumerate(dims))))

    pts = fetch_json(NUTS_PT_URL)
    rows = []
    for f in pts["features"]:
        nid = f["properties"]["id"]
        cc = EUROSTAT_TO_ISO.get(nid[:2], nid[:2])
        if cc not in EU or nid not in cat["geo"]:
            continue
        val = year = nace = None
        for n in ("C", "B-E"):  # manufacturing; fall back to total industry if C is missing
            for y in GVA_YEARS:
                v = get(freq="A", unit="CP_MEUR", nace_r2=n, geo=nid, time=y)
                if v is not None:
                    val, year, nace = v, y, n
                    break
            if val is not None:
                break
        if val is None:
            continue
        if is_offshore(nid):
            continue
        lon, lat = f["geometry"]["coordinates"]
        rows.append({"nuts3": nid, "name": f["properties"]["na"], "country_code": cc,
                     "lat": lat, "lon": lon, "gva_meur": val, "year": year, "nace": nace})
    df = pd.DataFrame(rows)

    # regional GDP (Eurostat nama_10r_3gdp, million EUR)
    g = fetch_json(GDP_URL)
    gdims, gsize = g["id"], g["size"]
    gcat = {k: g["dimension"][k]["category"]["index"] for k in gdims}
    gstr = [1] * len(gsize)
    for i in range(len(gsize) - 2, -1, -1):
        gstr[i] = gstr[i + 1] * gsize[i + 1]

    def gdp(nid):
        if nid not in gcat["geo"]:
            return None
        for y in GVA_YEARS:
            if y not in gcat["time"]:
                continue
            key = {"freq": "A", "unit": "MIO_EUR", "geo": nid, "time": y}
            v = g["value"].get(str(sum(gcat[k][key[k]] * gstr[i] for i, k in enumerate(gdims))))
            if v is not None:
                return v
        return None

    df["gdp_meur"] = [gdp(n) for n in df.nuts3]
    print(f"GDP found for {df.gdp_meur.notna().sum()} of {len(df)} regions")
    df.to_csv(out / "manufacturing_nuts3.csv", index=False)
    print(f"{len(df)} NUTS-3 regions with manufacturing GVA; "
          f"{(df.nace == 'B-E').sum()} use industry (B-E) fallback; years: {df.year.value_counts().to_dict()}")


EUROSTAT = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/"


def flat_by_geo_time(d):
    """Flatten a JSON-stat result whose only multi-valued dimensions are geo and time."""
    geo = d["dimension"]["geo"]["category"]["index"]
    times = list(d["dimension"]["time"]["category"]["index"])
    nt = len(times)
    return {g: {times[i]: d["value"].get(str(gi * nt + i)) for i in range(nt)} for g, gi in geo.items()}


def latest(series):
    for y in sorted(series, reverse=True):
        if series[y] is not None:
            return series[y], y
    return None, None


def build_productivity(out):
    """Nominal GDP per hour worked (EUR) per country = GDP (nama_10_gdp) / hours worked (nama_10_a10_e)."""
    gdp = flat_by_geo_time(fetch_json(EUROSTAT + "nama_10_gdp?format=JSON&lang=EN&na_item=B1GQ&unit=CP_MEUR&sinceTimePeriod=2023"))
    hrs = flat_by_geo_time(fetch_json(EUROSTAT + "nama_10_a10_e?format=JSON&lang=EN&nace_r2=TOTAL&na_item=EMP_DC&unit=THS_HW&sinceTimePeriod=2023"))
    rows = []
    for cc in EU:
        es = {"GR": "EL"}.get(cc, cc)
        for y in ("2025", "2024", "2023"):
            g, h = gdp.get(es, {}).get(y), hrs.get(es, {}).get(y)
            if g and h:
                rows.append({"country_code": cc, "gdp_per_hour_eur": round(g * 1e6 / (h * 1e3), 2), "prod_year": int(y)})
                break
    df = pd.DataFrame(rows)
    df.to_csv(out / "productivity.csv", index=False)
    print(f"productivity for {len(df)} countries")


def build_talent(out, regions):
    """Employment in high and medium-high tech manufacturing (htec_emp_reg2, NUTS-2, thousand persons).
    NUTS-2 location = GDP-weighted mean of its NUTS-3 label points."""
    d = fetch_json(EUROSTAT + "htec_emp_reg2?format=JSON&lang=EN&nace_r2=C_HTC_MH&unit=THS_PER&sex=T&sinceTimePeriod=2022")
    vals = flat_by_geo_time(d)
    regions = regions.assign(nuts2=regions.nuts3.str[:4])
    pts = regions.groupby("nuts2").apply(
        lambda g: pd.Series({"lat": np.average(g.lat, weights=g.gdp_meur),
                             "lon": np.average(g.lon, weights=g.gdp_meur)}), include_groups=False)
    rows = []
    for n2, p in pts.iterrows():
        v, y = latest(vals.get(n2, {}))
        if v is not None:
            rows.append({"nuts2": n2, "lat": p.lat, "lon": p.lon, "hitech_mfg_ths": v, "year": int(y)})
    df = pd.DataFrame(rows)
    df.to_csv(out / "talent_nuts2.csv", index=False)
    print(f"talent for {len(df)} of {len(pts)} NUTS-2 regions; total {df.hitech_mfg_ths.sum():,.0f}k persons")

    # spread each NUTS-2 total over its NUTS-3 regions in proportion to manufacturing GVA,
    # so talent sits where industry is instead of at one central point per NUTS-2 region
    r = regions.drop(columns="year").assign(share=regions.gva_meur / regions.groupby("nuts2").gva_meur.transform("sum"))
    t3 = r.merge(df[["nuts2", "hitech_mfg_ths", "year"]], on="nuts2", how="inner")
    t3["hitech_mfg_ths"] = t3.hitech_mfg_ths * t3.share
    t3[["nuts3", "name", "lat", "lon", "hitech_mfg_ths", "year"]].to_csv(out / "talent_nuts3.csv", index=False)
    print(f"talent spread over {len(t3)} NUTS-3 regions")


def build_firms_labour(out, regions):
    """Eurostat sbs_r_nuts2021, manufacturing (NACE C), NUTS-2:
    LOC_NR = number of local units (firms/plants); LC_EMP_LOC_TEUR = labour cost per person employed (k EUR/yr)."""
    d = fetch_json(EUROSTAT + "sbs_r_nuts2021?format=JSON&lang=EN&nace_r2=C&indic_sbs=LOC_NR"
                   "&indic_sbs=LC_EMP_LOC_TEUR&sinceTimePeriod=2021")
    dims, size = d["id"], d["size"]
    cat = {k: d["dimension"][k]["category"]["index"] for k in dims}
    st = [1] * len(size)
    for i in range(len(size) - 2, -1, -1):
        st[i] = st[i + 1] * size[i + 1]

    def series(ind, geo):
        if geo not in cat["geo"]:
            return {}
        return {t: d["value"].get(str(sum(cat[k][v] * st[i] for i, (k, v) in enumerate(
            zip(dims, ("A", "C", ind, geo, t)))))) for t in cat["time"]}

    regions = regions.assign(nuts2=regions.nuts3.str[:4])
    pts = regions.groupby("nuts2").apply(
        lambda g: pd.Series({"lat": np.average(g.lat, weights=g.gdp_meur),
                             "lon": np.average(g.lon, weights=g.gdp_meur)}), include_groups=False)
    rows = []
    for n2, p in pts.iterrows():
        firms, fy = latest(series("LOC_NR", n2))
        lc, ly = latest(series("LC_EMP_LOC_TEUR", n2))
        rows.append({"nuts2": n2, "country_code": EUROSTAT_TO_ISO.get(n2[:2], n2[:2]), "lat": p.lat, "lon": p.lon,
                     "mfg_firms": firms, "firms_year": fy, "mfg_labour_cost_teur": lc, "lc_year": ly})
    df = pd.DataFrame(rows)
    df.to_csv(out / "firms_labour_nuts2.csv", index=False)
    print(f"firms for {df.mfg_firms.notna().sum()}, regional labour cost for "
          f"{df.mfg_labour_cost_teur.notna().sum()} of {len(df)} NUTS-2 regions")


def main():
    out = Path(__file__).parent / "data"
    out.mkdir(exist_ok=True)
    build_manufacturing(out)
    build_productivity(out)
    build_talent(out, pd.read_csv(out / "manufacturing_nuts3.csv"))
    build_firms_labour(out, pd.read_csv(out / "manufacturing_nuts3.csv"))

    raw = [
        c for c in geonamescache.GeonamesCache().get_cities().values()
        if c["countrycode"] in EU and c["population"] > MIN_POP
    ]
    raw.sort(key=lambda c: -c["population"])

    kept = []
    for c in raw:
        if all(km(c["latitude"], c["longitude"], k["latitude"], k["longitude"]) > MERGE_KM for k in kept):
            kept.append(c)

    # drop cities not on the European mainland: assign each city to its nearest NUTS-3 label point
    pts = fetch_json(NUTS_PT_URL)["features"]
    pts = [(f["properties"]["id"], f["geometry"]["coordinates"]) for f in pts
           if EUROSTAT_TO_ISO.get(f["properties"]["id"][:2], f["properties"]["id"][:2]) in EU]

    def nearest_nuts(c):
        return min(pts, key=lambda p: km(c["latitude"], c["longitude"], p[1][1], p[1][0]))[0]

    before = len(kept)
    kept = [c for c in kept if c["countrycode"] not in OFFSHORE_COUNTRIES and not is_offshore(nearest_nuts(c))]
    print(f"removed {before - len(kept)} non-mainland cities")

    df = pd.DataFrame(
        {
            "city": [c["name"] for c in kept],
            "country_code": [c["countrycode"] for c in kept],
            "country": [COUNTRY_NAMES[c["countrycode"]] for c in kept],
            "lat": [c["latitude"] for c in kept],
            "lon": [c["longitude"] for c in kept],
            "population": [c["population"] for c in kept],
            "nuts2": [nearest_nuts(c)[:4] for c in kept],
        }
    )
    df.to_csv(out / "cities.csv", index=False)

    lc = pd.DataFrame(
        {
            "country_code": list(LABOUR_COST_EUR_H),
            "labour_cost_eur_h": list(LABOUR_COST_EUR_H.values()),
            "year": [LABOUR_YEAR[c] for c in LABOUR_COST_EUR_H],
        }
    )
    lc.to_csv(out / "labour_cost.csv", index=False)
    print(f"{len(df)} cities, {len(lc)} labour cost rows written to {out}")


if __name__ == "__main__":
    main()
