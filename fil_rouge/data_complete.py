"""Chargement et préparation des données d'échecs pour les exercices 1 et 2.

Arborescence attendue :
projet/
├── data/
│   └── echecs/
│       ├── parties.csv
│       └── coups_juges.csv
└── fil_rouge/
    └── data.py

Exécution depuis la racine du projet :
    uv run python fil_rouge/data.py
"""

from pathlib import Path

import numpy as np
import pandas as pd


# Le fichier est supposé se trouver dans fil_rouge/data.py.
DATA = Path(__file__).resolve().parents[1] / "data" / "echecs"

COLONNES_PARTIES = [
    "partie_id",
    "date_jeu",
    "cadence",
    "couleur",
    "resultat",
    "elo_joueur",
    "elo_adverse",
    "ouverture_famille",
    "gaffes_joueur",
    "erreurs_joueur",
    "imprecisions_joueur",
]

COLONNES_NUMERIQUES_PARTIES = [
    "elo_joueur",
    "elo_adverse",
    "gaffes_joueur",
    "erreurs_joueur",
    "imprecisions_joueur",
]

COLONNES_COUPS = [
    "partie_id",
    "ply",
    "numero_coup",
    "couleur",
    "coup",
    "phase",
    "eval_cp",
    "variation_eval_cp",
    "jugement",
]

CORRESPONDANCE_RESULTATS = {
    "win": "victoire",
    "draw": "nulle",
    "lose": "défaite",
}

SCORES_RESULTATS = {
    "victoire": 1.0,
    "nulle": 0.5,
    "défaite": 0.0,
}

# Seuils de perte des chances de gain décrits dans conseils.md.
SEUILS_JUGEMENT = {
    "imprécision": 0.10,
    "erreur": 0.20,
    "gaffe": 0.30,
}


def _verifier_fichier(chemin: Path) -> None:
    """Vérifie la présence du CSV et affiche un message exploitable."""
    if not chemin.is_file():
        raise FileNotFoundError(
            f"Fichier introuvable : {chemin}\n"
            "Vérifiez que le dossier data/echecs contient les fichiers CSV "
            "et que data.py se trouve dans le dossier fil_rouge."
        )


def _verifier_numeriques(df: pd.DataFrame, colonnes: list[str], fichier: str) -> None:
    """Refuse les colonnes numériques importées comme du texte.

    Les valeurs manquantes sont autorisées pour les Elo, conformément au LISEZMOI.
    """
    for colonne in colonnes:
        if not pd.api.types.is_numeric_dtype(df[colonne]):
            raise ValueError(
                f"La colonne '{colonne}' de {fichier} n'est pas numérique "
                f"(type détecté : {df[colonne].dtype}). "
                "Vérifiez notamment le séparateur et le format du CSV."
            )


def valider(df: pd.DataFrame) -> pd.DataFrame:
    """Valide la structure du CSV des parties et renvoie le DataFrame.

    Contrôles :
    - présence des colonnes nécessaires ;
    - colonnes obligatoires non entièrement vides (cadence et Elo exceptés) ;
    - types numériques ;
    - identifiants uniques ;
    - résultats et couleurs connus.
    """
    df = df.copy()

    manquantes = [col for col in COLONNES_PARTIES if col not in df.columns]
    if manquantes:
        raise ValueError(f"Colonnes absentes dans parties.csv : {manquantes}")

    # La cadence et les Elo peuvent légitimement être manquants.
    colonnes_vides_interdites = [
        col
        for col in [
            "partie_id",
            "date_jeu",
            "couleur",
            "resultat",
            "gaffes_joueur",
            "erreurs_joueur",
            "imprecisions_joueur",
        ]
        if df[col].isna().all()
    ]
    if colonnes_vides_interdites:
        raise ValueError(
            "Colonnes obligatoires entièrement vides dans parties.csv : "
            f"{colonnes_vides_interdites}"
        )

    _verifier_numeriques(df, COLONNES_NUMERIQUES_PARTIES, "parties.csv")

    if df["partie_id"].isna().any():
        raise ValueError("La colonne partie_id contient des valeurs manquantes.")

    doublons = df.loc[df["partie_id"].duplicated(keep=False), "partie_id"].unique()
    if len(doublons):
        raise ValueError(f"Identifiants de partie en double : {doublons[:10].tolist()}")

    resultats_valides = set(CORRESPONDANCE_RESULTATS)
    resultats_presents = set(df["resultat"].dropna().unique())
    resultats_inconnus = resultats_presents - resultats_valides
    if df["resultat"].isna().any() or resultats_inconnus:
        raise ValueError(
            "Résultats manquants ou inconnus dans parties.csv. "
            f"Valeurs inconnues : {sorted(resultats_inconnus)}"
        )

    couleurs_valides = {"white", "black"}
    couleurs_inconnues = set(df["couleur"].dropna().unique()) - couleurs_valides
    if df["couleur"].isna().any() or couleurs_inconnues:
        raise ValueError(
            "Couleurs manquantes ou inconnues dans parties.csv. "
            f"Valeurs inconnues : {sorted(couleurs_inconnues)}"
        )

    return df


def load_data() -> pd.DataFrame:
    """Charge et prépare une ligne par partie depuis parties.csv.

    Colonnes ajoutées/conservées pour les exercices suivants :
    - date_jeu : datetime naïf, conservé en UTC ;
    - resultat : victoire, nulle ou défaite ;
    - score : 1, 0.5 ou 0 ;
    - annee : année de la partie ;
    - heure : heure UTC ;
    - ecart_elo : Elo adverse moins Elo joueur ;
    - cadence : les valeurs manquantes deviennent « aucune ».

    Les Elo manquants restent manquants : ils ne sont pas remplacés artificiellement.
    """
    chemin = DATA / "parties.csv"
    _verifier_fichier(chemin)

    df = pd.read_csv(chemin)
    df = valider(df)

    # Les dates fournies sont en UTC sans suffixe de fuseau.
    df["date_jeu"] = pd.to_datetime(df["date_jeu"], errors="coerce")
    dates_invalides = df["date_jeu"].isna()
    if dates_invalides.any():
        exemples = df.loc[dates_invalides, "partie_id"].head(5).tolist()
        raise ValueError(
            f"{int(dates_invalides.sum())} date(s) invalides dans parties.csv. "
            f"Exemples de partie_id : {exemples}"
        )

    df["resultat"] = df["resultat"].map(CORRESPONDANCE_RESULTATS)
    df["score"] = df["resultat"].map(SCORES_RESULTATS)

    df["annee"] = df["date_jeu"].dt.year.astype("Int64")
    # Heure UTC (0 à 23), et non heure locale de Paris.
    df["heure"] = df["date_jeu"].dt.hour.astype("Int64")

    # Définition demandée : Elo adverse - Elo du joueur.
    # Si l'un des deux Elo manque, l'écart reste NaN.
    df["ecart_elo"] = df["elo_adverse"] - df["elo_joueur"]

    # Les 48 parties sans cadence doivent rester filtrables.
    df["cadence"] = df["cadence"].fillna("aucune").astype(str).str.strip()
    df.loc[df["cadence"].eq(""), "cadence"] = "aucune"

    return df


def load_coups() -> pd.DataFrame:
    """Charge les coups jugés mauvais : une ligne par faute du joueur.

    Attention : ce fichier ne contient ni les bons coups du joueur,
    ni les coups de l'adversaire. Il sert à étudier les fautes, pas à
    recalculer une précision globale.
    """
    chemin = DATA / "coups_juges.csv"
    _verifier_fichier(chemin)

    df = pd.read_csv(chemin)

    manquantes = [col for col in COLONNES_COUPS if col not in df.columns]
    if manquantes:
        raise ValueError(f"Colonnes absentes dans coups_juges.csv : {manquantes}")

    if df.empty:
        raise ValueError("coups_juges.csv ne contient aucune ligne.")

    colonnes_numeriques = [
        "ply",
        "numero_coup",
        "eval_cp",
        "variation_eval_cp",
    ]
    _verifier_numeriques(df, colonnes_numeriques, "coups_juges.csv")

    resultats_valides = {"inaccuracy", "mistake", "blunder"}
    inconnus = set(df["jugement"].dropna().unique()) - resultats_valides
    if df["jugement"].isna().any() or inconnus:
        raise ValueError(
            "Jugements manquants ou inconnus dans coups_juges.csv : "
            f"{sorted(inconnus)}"
        )

    phases_valides = {"opening", "middlegame", "endgame"}
    phases_inconnues = set(df["phase"].dropna().unique()) - phases_valides
    if df["phase"].isna().any() or phases_inconnues:
        raise ValueError(
            "Phases manquantes ou inconnues dans coups_juges.csv : "
            f"{sorted(phases_inconnues)}"
        )

    couleurs_inconnues = set(df["couleur"].dropna().unique()) - {"white", "black"}
    if df["couleur"].isna().any() or couleurs_inconnues:
        raise ValueError(
            "Couleurs manquantes ou inconnues dans coups_juges.csv : "
            f"{sorted(couleurs_inconnues)}"
        )

    return df


def chances_gain(cp: pd.Series | np.ndarray | float) -> pd.Series | np.ndarray | float:
    """Convertit une évaluation en centipions en chances de gain (-1 à +1).

    Formule indiquée dans conseils.md. Le centipion est borné à [-1000, 1000].
    Cette fonction attend une évaluation du point de vue du joueur étudié.
    """
    cp_borne = np.clip(cp, -1000, 1000)
    return 2 / (1 + np.exp(-0.00368208 * cp_borne)) - 1


def ajouter_indicateurs_coups(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute les évaluations avant/après et la gravité calculée des fautes.

    eval_cp est du point de vue des Blancs. Pour les Noirs, le signe est inversé.
    variation_eval_cp est la perte en centipions et reste positive.
    Les seuils de gravité sont ceux de conseils.md : 0.10, 0.20 et 0.30
    de chances de gain perdues.

    Cette estimation peut différer du jugement exporté pour les mats et les
    évaluations extrêmes ; le jugement fourni reste disponible dans 'jugement'.
    """
    requis = {"couleur", "eval_cp", "variation_eval_cp"}
    manquantes = requis - set(df.columns)
    if manquantes:
        raise ValueError(
            "Colonnes nécessaires absentes pour calculer les indicateurs : "
            f"{sorted(manquantes)}"
        )

    d = df.copy()
    signe = np.where(d["couleur"].eq("black"), -1, 1)

    # Évaluation après le coup, puis avant le coup, du point de vue du joueur.
    d["eval_apres_joueur_cp"] = d["eval_cp"] * signe
    d["eval_avant_joueur_cp"] = (
        d["eval_apres_joueur_cp"] + d["variation_eval_cp"]
    )

    d["chances_avant"] = chances_gain(d["eval_avant_joueur_cp"])
    d["chances_apres"] = chances_gain(d["eval_apres_joueur_cp"])
    d["perte_chances"] = d["chances_avant"] - d["chances_apres"]

    d["jugement_calcule"] = np.select(
        [
            d["perte_chances"] >= SEUILS_JUGEMENT["gaffe"],
            d["perte_chances"] >= SEUILS_JUGEMENT["erreur"],
            d["perte_chances"] >= SEUILS_JUGEMENT["imprécision"],
        ],
        ["gaffe", "erreur", "imprécision"],
        default="pas de seuil atteint",
    )

    # Une grosse erreur est une erreur/gaffe avec perte >= 200 cp,
    # ou une faute liée à un mat, selon conseils.md.
    d["grosse_erreur"] = (
        d["jugement"].isin(["mistake", "blunder"])
        & (
            d["variation_eval_cp"].ge(200)
            | d.get("mecanisme", pd.Series("", index=d.index)).eq("mate")
        )
    )

    return d


def familles(df: pd.DataFrame, min_n: int = 30) -> pd.DataFrame:
    """Calcule le score et le nombre moyen de gaffes par famille d'ouvertures.

    Les familles avec moins de min_n parties sont écartées.
    Le résultat est trié du score moyen le plus faible au plus élevé.
    Le nombre de parties est calculé avec nunique(partie_id), jamais avec len()
    après une jointure avec les coups.
    """
    if min_n < 1:
        raise ValueError("min_n doit être supérieur ou égal à 1.")

    colonnes = {
        "partie_id",
        "ouverture_famille",
        "score",
        "gaffes_joueur",
    }
    manquantes = colonnes - set(df.columns)
    if manquantes:
        raise ValueError(
            f"Colonnes absentes pour calculer les familles : {sorted(manquantes)}"
        )

    d = df.dropna(subset=["ouverture_famille"]).copy()

    agg = (
        d.groupby("ouverture_famille", as_index=False)
        .agg(
            n=("partie_id", "nunique"),
            score=("score", "mean"),
            gaffes=("gaffes_joueur", "mean"),
        )
        .rename(columns={"ouverture_famille": "famille"})
    )

    agg = agg.loc[agg["n"] >= min_n]
    return agg.sort_values(
        ["score", "gaffes", "n"],
        ascending=[True, False, False],
    ).reset_index(drop=True)


if __name__ == "__main__":
    parties = load_data()
    coups = load_coups()
    coups = ajouter_indicateurs_coups(coups)

    print("=== PARTIES ===")
    print(f"Dimensions : {parties.shape}")
    print(
        parties[
            [
                "partie_id",
                "date_jeu",
                "resultat",
                "score",
                "annee",
                "heure",
                "ecart_elo",
                "cadence",
            ]
        ].head()
    )

    print("\n=== RÉPARTITION DES RÉSULTATS ===")
    print(parties["resultat"].value_counts())
    print(f"\nScore moyen : {parties['score'].mean():.3f}")

    print("\n=== FAMILLES D'OUVERTURES À TRAVAILLER ===")
    print(familles(parties).head(10).to_string(index=False))

    print("\n=== COUPS JUGÉS ===")
    print(f"Dimensions : {coups.shape}")
    print(coups["jugement"].value_counts())
    print("\nFautes par phase :")
    print(coups.groupby("phase")["partie_id"].nunique().sort_values(ascending=False))

    print("\n=== GROSSES ERREURS ===")
    print(f"Nombre : {int(coups['grosse_erreur'].sum())}")
