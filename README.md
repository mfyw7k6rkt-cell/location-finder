# Location finder – Studio Intelligenza Artigianale (v1)

Scores EU cities (>50k inhabitants) as long-term production location on:
manufacturing market proximity (Eurostat GVA), 2-day delivery reach across the EU, shipping distance and labour cost.

## Run locally
    pip install -r requirements.txt
    streamlit run app.py

## Deploy (shareable link)
1. Push this folder to a GitHub repository.
2. Go to share.streamlit.io, "Create app", pick the repo, main file `app.py`.
3. Share the resulting URL.

## Update data
`python prepare_data.py` rebuilds `data/cities.csv` and `data/labour_cost.csv`
(needs `pip install geonamescache pandas`).

## Roadmap
1. Done: market and delivery reach weighted by Eurostat manufacturing GVA (nama_10r_3gva, NACE C, 2023).
2. Replace distance model with DHL Express transit times (MyDHL API) for the top 20.
3. Add industrial rent per city.
