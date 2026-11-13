"""Simule des positions non chevauchantes sur la série quotidienne HG=F."""

import argparse

import pandas as pd

from suivi import HORIZON, prix_cuivre, quantiles, signal


def simuler(prix, prevoir, frais):
    lignes = []
    disponible = 128
    # Décision avant la séance du mercredi, entrée à sa clôture.
    for i in range(128, len(prix) - HORIZON + 1):
        if i < disponible or prix.index[i].weekday() != 2:
            continue
        connu = float(prix.iloc[i - 1])
        if (prix.index[i].date() - prix.index[i - 1].date()).days > 3:
            continue
        bas, _, haut = prevoir(prix.iloc[i - 128:i])[-1]
        decision = signal(connu, bas, haut, frais)
        sens = {"achat": 1, "vente": -1, "aucune position": 0}[decision]
        entree = float(prix.iloc[i])
        sortie = float(prix.iloc[i + HORIZON - 1])
        rendement = sens * (sortie / entree - 1) - abs(sens) * frais
        lignes.append({"date": prix.index[i], "decision": decision, "rendement_net": rendement})
        if sens:
            disponible = i + HORIZON
    return pd.DataFrame(lignes)


def main():
    from chronos import Chronos2Pipeline

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frais", type=float, default=0.002)
    args = parser.parse_args()
    if not 0 <= args.frais < 1:
        parser.error("--frais doit être compris entre 0 et 1.")
    base = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cpu")
    resultats = simuler(prix_cuivre(), lambda x: quantiles(base, x, HORIZON), args.frais)
    if resultats.empty:
        raise ValueError("Aucune fenêtre de simulation disponible.")
    resultats["capital"] = (1 + resultats["rendement_net"]).cumprod()
    capital = resultats["capital"]
    baisse = (capital / capital.cummax().clip(lower=1) - 1).min()
    resultats.to_csv("backtest_cuivre.csv", index=False)
    print(f"Positions : {resultats.decision.ne('aucune position').sum()}")
    print(f"Rendement net simulé : {capital.iloc[-1] - 1:.2%}")
    print(f"Baisse maximale entre clôtures de positions : {baisse:.2%}")


if __name__ == "__main__":
    main()
