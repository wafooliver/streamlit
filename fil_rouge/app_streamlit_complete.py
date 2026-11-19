"""Application Streamlit ChessXP : identifier les axes de progression aux échecs.

Lancement depuis la racine du projet :
    uv run streamlit run fil_rouge/app_streamlit.py --server.address 127.0.0.1

Arborescence attendue :
    projet/
    ├── data/echecs/parties.csv
    ├── data/echecs/coups_juges.csv
    └── fil_rouge/
        ├── data.py
        └── app_streamlit.py
"""

from datetime import date, timedelta

import pandas as pd
import plotly.express as px
import streamlit as st

from data import familles, load_coups, load_data


st.set_page_config(
    page_title="Mon entraînement aux échecs",
    page_icon="♟️",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Chargement : le cache évite de relire les CSV à chaque interaction.
# ---------------------------------------------------------------------------
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
    st.info(
        "Vérifiez que parties.csv et coups_juges.csv sont dans "
        "data/echecs/, et que data.py et app_streamlit.py sont dans fil_rouge/."
    )
    st.stop()


# ---------------------------------------------------------------------------
# Titre et filtres
# ---------------------------------------------------------------------------
st.title("Que dois-je travailler pour progresser aux échecs ?")
st.caption(
    "Analyse de mes parties, de mes résultats et des gaffes commises "
    "sur la période sélectionnée."
)

date_min = parties["date_jeu"].min().date()
date_max = parties["date_jeu"].max().date()
cadences_disponibles = sorted(parties["cadence"].dropna().unique().tolist())

with st.sidebar:
    st.header("Filtres")

    cadences = st.multiselect(
        "Cadence",
        options=cadences_disponibles,
        default=cadences_disponibles,
        key="filtre_cadence",
        help="« aucune » correspond notamment aux parties par correspondance ou sans pendule.",
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

# Le date_input renvoie temporairement une seule date tant que la deuxième
# borne n'a pas été choisie : on attend alors que la période soit complète.
if not isinstance(periode, (tuple, list)) or len(periode) != 2:
    st.info("Sélectionnez une date de début et une date de fin pour afficher l'analyse.")
    st.stop()

date_debut, date_fin = periode
if date_debut > date_fin:
    st.warning("La date de début doit être antérieure ou égale à la date de fin.")
    st.stop()

# Filtrage des parties. La borne supérieure est exclusive après ajout d'un jour
# pour inclure toutes les parties jouées le dernier jour.
vue = parties.loc[
    parties["cadence"].isin(cadences)
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


# ---------------------------------------------------------------------------
# Indicateurs principaux
# ---------------------------------------------------------------------------
nombre_parties = int(vue["partie_id"].nunique())
score_moyen = float(vue["score"].mean())
total_gaffes = int(vue["gaffes_joueur"].fillna(0).sum())
gaffes_par_partie = total_gaffes / nombre_parties if nombre_parties else 0.0

m1, m2, m3 = st.columns(3)
m1.metric("Parties jouées", f"{nombre_parties:,}".replace(",", " "))
m2.metric("Score moyen", f"{score_moyen:.1%}".replace(".", ","))
m3.metric("Gaffes par partie", f"{gaffes_par_partie:.2f}".replace(".", ","))

st.caption(
    f"Période : du {date_debut:%d/%m/%Y} au {date_fin:%d/%m/%Y} · "
    f"{len(cadences)} cadence(s) sélectionnée(s) · "
    f"{'toutes les couleurs' if couleur == 'Toutes' else couleur.lower()}"
)


# ---------------------------------------------------------------------------
# Figures calculées exclusivement sur les parties filtrées.
# ---------------------------------------------------------------------------
st.subheader("Mes principaux axes de travail")
col_gauche, col_droite = st.columns(2)

with col_gauche:
    st.markdown("**1. Familles d'ouvertures où mon score est le plus faible**")

    # Un minimum de 5 parties évite d'afficher des moyennes trop peu fiables
    # lorsque l'utilisateur réduit la période ou les cadences.
    stats_ouvertures = familles(vue, min_n=5)

    if stats_ouvertures.empty:
        st.info(
            "Pas assez de parties par famille d'ouverture avec ces filtres "
            "(minimum : 5 parties par famille)."
        )
    else:
        graphique_ouvertures = stats_ouvertures.head(10).sort_values(
            "score", ascending=True
        )
        fig_ouvertures = px.bar(
            graphique_ouvertures,
            x="score",
            y="famille",
            orientation="h",
            text=graphique_ouvertures["score"].map(lambda x: f"{x:.0%}"),
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

    ids_parties_filtrees = vue["partie_id"].dropna().unique()
    coups_filtres = coups.loc[
        coups["partie_id"].isin(ids_parties_filtrees)
    ].copy()

    # Dans les données ChessXP, une gaffe correspond au jugement "blunder".
    gaffes = coups_filtres.loc[coups_filtres["jugement"].eq("blunder")].copy()

    ordre_phases = ["opening", "middlegame", "endgame"]
    noms_phases = {
        "opening": "Ouverture",
        "middlegame": "Milieu de partie",
        "endgame": "Finale",
    }

    if gaffes.empty:
        st.info("Aucune gaffe recensée dans les parties correspondant aux filtres.")
    else:
        stats_phases = (
            gaffes.groupby("phase")
            .size()
            .reindex(ordre_phases, fill_value=0)
            .rename("nombre_gaffes")
            .reset_index()
        )
        stats_phases["phase"] = stats_phases["phase"].map(noms_phases)

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

# ---------------------------------------------------------------------------
# Lecture rapide des résultats
# ---------------------------------------------------------------------------
with st.expander("Comment lire ces indicateurs ?"):
    st.markdown(
        """
        - **Score moyen** : 1 point pour une victoire, 0,5 pour une nulle et
          0 pour une défaite. Il ne s'agit pas d'un pourcentage de précision.
        - **Gaffes par partie** : nombre de coups classés `blunder` divisé par
          le nombre de parties affichées.
        - **Ouvertures** : les familles affichées ont au moins 5 parties dans
          la sélection. Le score le plus faible apparaît en premier.
        - **Phases** : les gaffes sont réparties entre ouverture, milieu de
          partie et finale. Les données de coups ne contiennent que les coups
          jugés mauvais du joueur, pas ses bons coups ni ceux de l'adversaire.
        - Les dates sources sont enregistrées en UTC ; cette application
          conserve cette référence et ne convertit pas les heures.
        """
    )
