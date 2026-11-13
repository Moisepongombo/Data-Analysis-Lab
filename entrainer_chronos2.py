"""Entraîne un modèle Chronos-2 par série et mesure l'erreur de prévision."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import peft  # Exige PEFT pour éviter le repli vers un entraînement complet.
import torch
from chronos import Chronos2Pipeline


HORIZONS = {"PCOPPUSDM": 1, "INDPRO": 3}
CONTEXTE = 96


def lire_serie(jeu, identifiant):
    serie = jeu.loc[jeu["id"].eq(identifiant)].sort_values("timestamp")
    mois = pd.PeriodIndex(serie["timestamp"], freq="M")
    valeurs = serie["target"].to_numpy(dtype=np.float32)
    if not len(mois) or not np.isfinite(valeurs).all():
        raise ValueError(f"Série vide ou valeurs invalides pour {identifiant}.")
    if not mois.equals(pd.period_range(mois[0], mois[-1], freq="M")):
        raise ValueError(f"Mois manquants ou dupliqués pour {identifiant}.")
    return mois, valeurs


def comparer(base, adapte, valeurs, debut, fin, horizon, identifiant, periode):
    # À chaque origine, seuls les mois antérieurs sont donnés au modèle.
    origines = range(debut, fin - horizon + 1)
    contextes = [valeurs[max(0, i - CONTEXTE):i] for i in origines]
    reels = np.stack([valeurs[i:i + horizon] for i in origines])
    dernier = np.asarray([valeurs[i - 1] for i in origines])
    predictions = {"reference": np.repeat(dernier[:, None], horizon, axis=1)}
    if identifiant == "INDPRO":
        predictions["reference_saisonniere"] = np.stack(
            [valeurs[i - 12:i - 12 + horizon] for i in origines]
        )
    for nom, modele in (("chronos2", base), ("chronos2_lora", adapte)):
        quantiles, _ = modele.predict_quantiles(
            contextes,
            prediction_length=horizon,
            quantile_levels=[0.5],
            batch_size=8,
        )
        predictions[nom] = np.stack([q[0, :, 0].cpu().numpy() for q in quantiles])
    return [
        {
            "id": identifiant,
            "periode": periode,
            "modele": nom,
            "horizon": mois + 1,
            "n_previsions": len(reels),
            "mae": np.abs(prevision - reels).mean(axis=0)[mois],
        }
        for nom, prevision in predictions.items()
        for mois in range(horizon)
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decoupage", type=Path, default=Path("decoupage"))
    parser.add_argument("--sortie", type=Path, default=Path("resultats_chronos2"))
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--evaluer-test", action="store_true", help="À utiliser après avoir choisi les réglages sur la validation.")
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps doit être positif.")

    noms = ["entrainement", "validation"]
    if args.evaluer_test:
        noms.append("test")
    jeux = [
        pd.read_csv(args.decoupage / f"{nom}.csv", parse_dates=["timestamp"])
        for nom in noms
    ]
    appareil = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Appareil : {appareil} | PEFT : {peft.__version__}", flush=True)
    base = Chronos2Pipeline.from_pretrained(
        "amazon/chronos-2", device_map=appareil, dtype=torch.float32
    )
    scores = []
    for identifiant, horizon in HORIZONS.items():
        blocs = [lire_serie(jeu, identifiant) for jeu in jeux]
        dates = blocs[0][0].append([mois for mois, _ in blocs[1:]])
        if not dates.equals(pd.period_range(dates[0], dates[-1], freq="M")):
            raise ValueError(f"Les périodes de {identifiant} doivent se suivre sans chevauchement.")
        if len(blocs[0][1]) < 24 + horizon or any(len(v) < horizon for _, v in blocs[1:]):
            raise ValueError(f"Historique insuffisant pour {identifiant}.")
        valeurs = np.concatenate([v for _, v in blocs])
        entrainement = blocs[0][1]
        print(f"Entraînement LoRA de {identifiant} à {horizon} mois", flush=True)
        adapte = base.fit(
            inputs=[entrainement],
            prediction_length=horizon,
            finetune_mode="lora",
            context_length=CONTEXTE,
            min_past=24,
            num_steps=args.steps,
            batch_size=4,
            learning_rate=1e-4,
            output_dir=args.sortie / identifiant,
            optim="adamw_torch",
            seed=42,
            logging_steps=10,
        )
        debut = len(entrainement)
        for nom, (_, bloc) in zip(noms[1:], blocs[1:]):
            fin = debut + len(bloc)
            scores.extend(comparer(base, adapte, valeurs, debut, fin, horizon, identifiant, nom))
            debut = fin
        del adapte

    tableau = pd.DataFrame(scores)
    tableau.to_csv(args.sortie / "scores.csv", index=False)
    print(tableau.to_string(index=False))


if __name__ == "__main__":
    main()
