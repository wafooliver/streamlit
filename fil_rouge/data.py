"""Chargement et préparation des données du fil rouge."""

from pathlib import Path

import pandas as pd


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


def valider(df: pd.DataFrame) -> pd.DataFrame:
    """Valide la structure du fichier des parties."""
    df = df.copy()
    manquantes = [col for col in COLONNES_PARTIES if col not in df.columns]
    if manquantes:
        raise ValueError(f"Colonnes absentes dans parties.csv : {manquantes}")

    obligatoires = [
        "partie_id",
        "date_jeu",
        "couleur",
        "resultat",
        "gaffes_joueur",
        "erreurs_joueur",
        "imprecisions_joueur",
    ]
    colonnes_vides = [col for col in obligatoires if df[col].isna().all()]
    if colonnes_vides:
        raise ValueError(f"Colonnes obligatoires entièrement vides : {colonnes_vides}")

    for colonne in COLONNES_NUMERIQUES_PARTIES:
        if not pd.api.types.is_numeric_dtype(df[colonne]):
            raise ValueError(
                f"La colonne '{colonne}' de parties.csv n'est pas numérique "
                f"(type détecté : {df[colonne].dtype})."
            )

    if df["partie_id"].isna().any():
        raise ValueError("La colonne partie_id contient des valeurs manquantes.")

    doublons = df.loc[df["partie_id"].duplicated(keep=False), "partie_id"].unique()
    if len(doublons):
        raise ValueError(f"Identifiants de partie en double : {doublons[:10].tolist()}")

    resultats_inconnus = set(df["resultat"].dropna()) - set(CORRESPONDANCE_RESULTATS)
    if df["resultat"].isna().any() or resultats_inconnus:
        raise ValueError(
            "Résultats manquants ou inconnus dans parties.csv. "
            f"Valeurs inconnues : {sorted(resultats_inconnus)}"
        )

    couleurs_inconnues = set(df["couleur"].dropna()) - {"white", "black"}
    if df["couleur"].isna().any() or couleurs_inconnues:
        raise ValueError(
            "Couleurs manquantes ou inconnues dans parties.csv. "
            f"Valeurs inconnues : {sorted(couleurs_inconnues)}"
        )

    return df


def load_data() -> pd.DataFrame:
    """Charge une ligne par partie avec les colonnes utilisées par l'application."""
    chemin = DATA / "parties.csv"
    if not chemin.is_file():
        raise FileNotFoundError(f"Fichier introuvable : {chemin}")

    df = valider(pd.read_csv(chemin))
    df["date_jeu"] = pd.to_datetime(df["date_jeu"], errors="coerce")
    if df["date_jeu"].isna().any():
        ids = df.loc[df["date_jeu"].isna(), "partie_id"].head(5).tolist()
        raise ValueError(f"Dates invalides dans parties.csv. Exemples : {ids}")

    df["resultat"] = df["resultat"].map(CORRESPONDANCE_RESULTATS)
    df["score"] = df["resultat"].map(SCORES_RESULTATS)
    df["annee"] = df["date_jeu"].dt.year.astype("Int64")
    df["heure"] = df["date_jeu"].dt.hour.astype("Int64")
    df["ecart_elo"] = df["elo_adverse"] - df["elo_joueur"]
    df["cadence"] = df["cadence"].fillna("aucune").astype(str).str.strip()
    df.loc[df["cadence"].eq(""), "cadence"] = "aucune"
    return df


def load_coups() -> pd.DataFrame:
    """Charge les coups jugés mauvais du joueur."""
    chemin = DATA / "coups_juges.csv"
    if not chemin.is_file():
        raise FileNotFoundError(f"Fichier introuvable : {chemin}")

    df = pd.read_csv(chemin)
    requises = [
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
    manquantes = [col for col in requises if col not in df.columns]
    if manquantes:
        raise ValueError(f"Colonnes absentes dans coups_juges.csv : {manquantes}")
    if df.empty:
        raise ValueError("coups_juges.csv ne contient aucune ligne.")

    for colonne in ["ply", "numero_coup", "eval_cp", "variation_eval_cp"]:
        if not pd.api.types.is_numeric_dtype(df[colonne]):
            raise ValueError(
                f"La colonne '{colonne}' de coups_juges.csv n'est pas numérique."
            )

    jugements_valides = {"inaccuracy", "mistake", "blunder"}
    jugements_inconnus = set(df["jugement"].dropna()) - jugements_valides
    if df["jugement"].isna().any() or jugements_inconnus:
        raise ValueError(
            f"Jugements invalides dans coups_juges.csv : {jugements_inconnus}"
        )

    phases_valides = {"opening", "middlegame", "endgame"}
    phases_inconnues = set(df["phase"].dropna()) - phases_valides
    if df["phase"].isna().any() or phases_inconnues:
        raise ValueError(f"Phases invalides dans coups_juges.csv : {phases_inconnues}")

    couleurs_inconnues = set(df["couleur"].dropna()) - {"white", "black"}
    if df["couleur"].isna().any() or couleurs_inconnues:
        raise ValueError(f"Couleurs invalides dans coups_juges.csv : {couleurs_inconnues}")

    return df


def familles(df: pd.DataFrame, min_n: int = 30) -> pd.DataFrame:
    """Score moyen et gaffes moyennes par famille d'ouverture."""
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
        raise ValueError(f"Colonnes absentes pour calculer les familles : {manquantes}")

    stats = (
        df.dropna(subset=["ouverture_famille"])
        .groupby("ouverture_famille", as_index=False)
        .agg(
            n=("partie_id", "nunique"),
            score=("score", "mean"),
            gaffes=("gaffes_joueur", "mean"),
        )
        .rename(columns={"ouverture_famille": "famille"})
    )
    stats = stats.loc[stats["n"] >= min_n]
    return stats.sort_values(
        ["score", "gaffes", "n"],
        ascending=[True, False, False],
    ).reset_index(drop=True)


if __name__ == "__main__":
    parties = load_data()
    coups = load_coups()
    print(parties.shape)
    print(coups.shape)
