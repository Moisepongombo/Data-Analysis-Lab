"""Crée les jeux chronologiques pour l'évaluation de Chronos-2."""

import argparse
from pathlib import Path

import pandas as pd


DEBUT_VALIDATION = "2022-01-01"
DEBUT_TEST = "2024-01-01"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("entree", nargs="?", type=Path, default=Path("donnees_nettoyees_chronos2.csv"))
    parser.add_argument("-o", "--sortie", type=Path, default=Path("decoupage"))
    args = parser.parse_args()

    donnees = pd.read_csv(args.entree, usecols=["id", "timestamp", "target"])
    donnees["timestamp"] = pd.to_datetime(donnees["timestamp"], format="%Y-%m-%d")
    if donnees.empty or donnees.isna().any().any():
        parser.error("Le CSV nettoyé doit contenir des données sans valeurs manquantes.")
    if donnees.duplicated(["id", "timestamp"]).any():
        parser.error("Chaque série doit avoir une seule observation par date.")
    donnees = donnees[["id", "timestamp", "target"]].sort_values(["id", "timestamp"])
    dates = donnees["timestamp"]
    series = set(donnees["id"])
    jeux = {
        "entrainement": donnees[dates < DEBUT_VALIDATION],
        "validation": donnees[(dates >= DEBUT_VALIDATION) & (dates < DEBUT_TEST)],
        "test": donnees[dates >= DEBUT_TEST],
    }

    for nom, jeu in jeux.items():
        if set(jeu["id"]) != series:
            parser.error(f"Il manque au moins une série dans le jeu {nom}.")
        if (args.sortie / f"{nom}.csv").resolve() == args.entree.resolve():
            parser.error("Le fichier d'entrée ne doit pas servir de fichier de sortie.")

    args.sortie.mkdir(parents=True, exist_ok=True)
    for nom, jeu in jeux.items():
        chemin = args.sortie / f"{nom}.csv"
        jeu.to_csv(chemin, index=False, date_format="%Y-%m-%d")
        effectifs = jeu.groupby("id").size().to_dict()
        print(f"{chemin} | {effectifs}")


if __name__ == "__main__":
    main()
