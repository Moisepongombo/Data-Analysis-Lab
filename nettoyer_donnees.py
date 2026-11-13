"""Convertit l'export FRED du cuivre et de la production industrielle pour Chronos-2."""

import argparse
from pathlib import Path

import pandas as pd


SERIES = ["PCOPPUSDM", "INDPRO"]


def nettoyer(brut):
    donnees = brut[["observation_date", *SERIES]].copy()
    donnees["observation_date"] = pd.to_datetime(
        donnees["observation_date"], format="%Y-%m-%d", errors="coerce"
    )
    for colonne in SERIES:
        valeurs = pd.to_numeric(donnees[colonne], errors="coerce")
        donnees[colonne] = valeurs.replace([float("inf"), float("-inf")], float("nan"))

    # Le cuivre et la production industrielle commencent à des dates différentes.
    propre = (
        donnees.rename(columns={"observation_date": "timestamp"})
        .melt(id_vars="timestamp", value_vars=SERIES, var_name="id", value_name="target")
        .dropna(subset=["timestamp", "target"])
        .drop_duplicates()
        .sort_values(["id", "timestamp"])
        .reset_index(drop=True)
    )

    doublons = propre[propre.duplicated(["id", "timestamp"], keep=False)]
    if not doublons.empty:
        raise ValueError(f"Valeurs contradictoires :\n{doublons.to_string(index=False)}")

    for nom in SERIES:
        dates = propre.loc[propre["id"].eq(nom), "timestamp"]
        if dates.empty or not dates.dt.is_month_start.all():
            raise ValueError(f"{nom} : série vide ou date hors début de mois.")
        mois = pd.PeriodIndex(dates, freq="M")
        attendus = pd.period_range(mois.min(), mois.max(), freq="M")
        manquants = attendus.difference(mois)
        print(f"{nom} | {len(mois)} observations | {len(manquants)} mois manquants")
        if len(manquants):
            print(", ".join(manquants.astype(str)))

    return propre[["id", "timestamp", "target"]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("entree", nargs="?", type=Path, default=Path("donnees_brutes_FRED.csv"))
    parser.add_argument("-o", "--sortie", type=Path, default=Path("donnees_nettoyees_chronos2.csv"))
    args = parser.parse_args()
    if args.entree.resolve() == args.sortie.resolve():
        parser.error("Le fichier de sortie doit être différent du fichier brut.")

    try:
        brut = pd.read_csv(args.entree, usecols=["observation_date", *SERIES])
        propre = nettoyer(brut)
        propre.to_csv(args.sortie, index=False, date_format="%Y-%m-%d")
    except (OSError, ValueError) as erreur:
        parser.exit(1, f"{erreur}\n")

    print(f"CSV exporté vers {args.sortie}")


if __name__ == "__main__":
    main()
