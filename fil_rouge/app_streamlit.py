"""Application Streamlit du fil rouge : axes de progression aux échecs."""

from datetime import timedelta

import pandas as pd
import plotly.express as px
import streamlit as st

from data import familles, load_coups, load_data


st.set_page_config(
    page_title="Mon entraînement aux échecs",
    page_icon="♟️",
    layout="wide",
)


@st.cache_data(show_spinner="Chargement des parties…")
def charger_parties() -> pd.DataFrame:
    return load_data()


@st.cache_data(show_spinner="Chargement des coups analysés…")
def charger_coups() -> pd.DataFrame:
    return load_coups()


try:
    parties = charger_parties()
    coups = charger_coups()
except (FileNotFoundError, ValueError, KeyError) as erreur:
    st.error(f"Impossible de charger les données : {erreur}")
    st.stop()


st.title("Que dois-je travailler pour progresser aux échecs ?")
st.caption("Analyse des parties, des résultats et des gaffes sur la période choisie.")

date_min = parties["date_jeu"].min().date()
date_max = parties["date_jeu"].max().date()
cadences = sorted(parties["cadence"].dropna().unique().tolist())

with st.sidebar:
    st.header("Filtres")
    cadences_selectionnees = st.multiselect(
        "Cadence",
        options=cadences,
        default=cadences,
        key="filtre_cadence",
        help="« aucune » correspond notamment aux parties par correspondance.",
    )
    periode = st.date_input(
        "Période des parties",
        value=(date_min, date_max),
        min_value=date_min,
        max_value=date_max,
        format="DD/MM/YYYY",
        key="filtre_periode",
    )
    couleur = st.selectbox(
        "Couleur jouée",
        options=["Toutes", "Blancs", "Noirs"],
        index=0,
        key="filtre_couleur",
    )

if not isinstance(periode, (tuple, list)) or len(periode) != 2:
    st.info("Sélectionnez une date de début et une date de fin.")
    st.stop()

date_debut, date_fin = periode
if date_debut > date_fin:
    st.warning("La date de début doit être antérieure ou égale à la date de fin.")
    st.stop()

vue = parties.loc[
    parties["cadence"].isin(cadences_selectionnees)
    & (parties["date_jeu"] >= pd.Timestamp(date_debut))
    & (parties["date_jeu"] < pd.Timestamp(date_fin + timedelta(days=1)))
].copy()

if couleur == "Blancs":
    vue = vue.loc[vue["couleur"].eq("white")].copy()
elif couleur == "Noirs":
    vue = vue.loc[vue["couleur"].eq("black")].copy()

if vue.empty:
    st.warning("Aucune partie ne correspond à ces filtres. Modifiez votre sélection.")
    st.stop()

nombre_parties = int(vue["partie_id"].nunique())
score_moyen = float(vue["score"].mean())
gaffes_par_partie = float(vue["gaffes_joueur"].fillna(0).sum()) / nombre_parties

col1, col2, col3 = st.columns(3)
col1.metric("Parties jouées", f"{nombre_parties:,}".replace(",", " "))
col2.metric("Score moyen", f"{score_moyen:.1%}".replace(".", ","))
col3.metric("Gaffes par partie", f"{gaffes_par_partie:.2f}".replace(".", ","))

st.caption(
    f"Période : du {date_debut:%d/%m/%Y} au {date_fin:%d/%m/%Y} · "
    f"{len(cadences_selectionnees)} cadence(s) sélectionnée(s)"
)

st.subheader("Mes principaux axes de travail")
col_gauche, col_droite = st.columns(2)

with col_gauche:
    st.markdown("**1. Familles d'ouvertures où mon score est le plus faible**")
    stats_ouvertures = familles(vue, min_n=5)

    if stats_ouvertures.empty:
        st.info("Pas assez de parties par famille d'ouverture (minimum : 5).")
    else:
        graphique = stats_ouvertures.head(10).sort_values("score", ascending=True)
        fig_ouvertures = px.bar(
            graphique,
            x="score",
            y="famille",
            orientation="h",
            text=graphique["score"].map(lambda valeur: f"{valeur:.0%}"),
            hover_data={
                "n": True,
                "gaffes": ":.2f",
                "score": ":.1%",
                "famille": False,
            },
            labels={
                "score": "Score moyen",
                "famille": "Famille d'ouverture",
                "n": "Nombre de parties",
                "gaffes": "Gaffes par partie",
            },
            title="Les ouvertures à retravailler",
        )
        fig_ouvertures.update_xaxes(tickformat=".0%", range=[0, 1])
        fig_ouvertures.update_traces(textposition="outside", cliponaxis=False)
        fig_ouvertures.update_layout(
            height=430,
            margin=dict(l=10, r=25, t=55, b=35),
            yaxis_title=None,
            xaxis_title="Score moyen",
        )
        st.plotly_chart(fig_ouvertures, use_container_width=True)

with col_droite:
    st.markdown("**2. Phases dans lesquelles je commets le plus de gaffes**")
    ids_parties = vue["partie_id"].dropna().unique()
    coups_filtres = coups.loc[coups["partie_id"].isin(ids_parties)].copy()
    gaffes = coups_filtres.loc[coups_filtres["jugement"].eq("blunder")].copy()

    if gaffes.empty:
        st.info("Aucune gaffe recensée dans les parties correspondant aux filtres.")
    else:
        ordre = ["opening", "middlegame", "endgame"]
        noms = {
            "opening": "Ouverture",
            "middlegame": "Milieu de partie",
            "endgame": "Finale",
        }
        stats_phases = (
            gaffes.groupby("phase")
            .size()
            .reindex(ordre, fill_value=0)
            .rename("nombre_gaffes")
            .reset_index()
        )
        stats_phases["phase"] = stats_phases["phase"].map(noms)

        fig_phases = px.bar(
            stats_phases,
            x="phase",
            y="nombre_gaffes",
            text="nombre_gaffes",
            labels={
                "phase": "Phase de jeu",
                "nombre_gaffes": "Nombre de gaffes",
            },
            title="Répartition de mes gaffes par phase",
        )
        fig_phases.update_traces(textposition="outside")
        fig_phases.update_layout(
            height=430,
            margin=dict(l=10, r=15, t=55, b=35),
            xaxis_title=None,
            yaxis_title="Nombre de gaffes",
        )
        st.plotly_chart(fig_phases, use_container_width=True)

with st.expander("Comment lire ces indicateurs ?"):
    st.markdown(
        """
        - **Score moyen** : 1 pour une victoire, 0,5 pour une nulle, 0 pour une défaite.
        - **Gaffes par partie** : nombre de coups classés `blunder` dans l'export,
          divisé par le nombre de parties affichées.
        - **Ouvertures** : seules les familles avec au moins 5 parties sont affichées.
        - **Phases** : les gaffes sont comptées uniquement parmi les coups analysés
          du joueur dans les parties sélectionnées.
        """
    )
