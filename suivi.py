"""Collecte quotidienne et signaux simulés de Data Analysis Lab."""

import argparse
import json
import math
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

import pandas as pd

SERIES = {"INDPRO": 3, "PCOPPUSDM": 1, "CPIAUCSL": 1}
FRED = "https://fred.stlouisfed.org/graph/fredgraph.csv"
RSS = "https://www.federalreserve.gov/feeds/press_monetary.xml"
ACTIF = "HG=F"
HORIZON = 20


def signal(dernier, bas, haut, frais):
    if bas > dernier * (1 + frais):
        return "achat"
    if haut < dernier * (1 - frais):
        return "vente"
    return "aucune position"


def prix_cuivre():
    import yfinance as yf

    prix = yf.Ticker(ACTIF).history(period="5y", auto_adjust=False)["Close"].dropna()
    # Exclut la séance courante pour éviter une clôture encore provisoire.
    aujourd_hui = pd.Timestamp.now(tz="America/New_York").date()
    prix = prix[prix.index.date < aujourd_hui]
    if len(prix) < 128 or not prix.index.is_unique:
        raise ValueError("Historique quotidien du cuivre insuffisant ou dupliqué.")
    return prix.sort_index()


def quantiles(modele, valeurs, horizon):
    resultat, _ = modele.predict_quantiles(
        [valeurs.to_numpy(dtype="float32")],
        prediction_length=horizon,
        quantile_levels=[0.1, 0.5, 0.9],
    )
    return resultat[0][0].cpu().tolist()


def actualiser(sortie, adaptateur, frais):
    from chronos import Chronos2Pipeline

    maintenant = datetime.now(ZoneInfo("America/Toronto"))
    fichier = sortie / f"{maintenant:%Y-%m-%d}.json"
    if fichier.exists():
        print(f"Collecte déjà enregistrée : {fichier}")
        return

    base = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cpu")
    observations = pd.read_csv(FRED + "?id=" + ",".join(SERIES), parse_dates=["observation_date"])
    observations = observations.set_index("observation_date").sort_index()
    bilan = {"collecte_le": maintenant.isoformat(), "fred": {}, "actualites": [], "erreurs": {}}
    for code, horizon in SERIES.items():
        disponible = pd.to_numeric(observations[code], errors="coerce").dropna()
        if len(disponible) < 2:
            bilan["erreurs"][code] = "Historique mensuel insuffisant"
            continue
        dates = pd.date_range(end=disponible.index[-1], periods=96, freq="MS")
        # Chronos-2 masque les NaN et conserve la position de chaque mois.
        serie = disponible.reindex(dates)
        modele = base
        if code == "INDPRO" and adaptateur:
            modele = Chronos2Pipeline.from_pretrained(str(adaptateur), device_map="cpu")
        mois = pd.period_range(serie.index[-1].to_period("M") + 1, periods=horizon, freq="M")
        bilan["fred"][code] = {
            "source": FRED + "?id=" + code,
            "modele": "chronos2_lora" if modele is not base else "chronos2",
            "observations": {
                d.strftime("%Y-%m"): None if pd.isna(v) else float(v)
                for d, v in serie.items()
            },
            "previsions": [
                {"mois": str(d), "q10": q[0], "q50": q[1], "q90": q[2]}
                for d, q in zip(mois, quantiles(modele, serie, horizon))
            ],
        }

    try:
        with urlopen(Request(RSS, headers={"User-Agent": "DataAnalysisLab/1.0"}), timeout=20) as rep:
            flux = ET.fromstring(rep.read())
        bilan["actualites"] = [
            {"titre": item.findtext("title"), "url": item.findtext("link"),
             "publie_le": item.findtext("pubDate"), "source": RSS}
            for item in flux.findall("./channel/item")[:10]
        ]
    except (OSError, ET.ParseError) as erreur:
        bilan["erreurs"]["actualites"] = str(erreur)

    try:
        prix = prix_cuivre()
        dernier = float(prix.iloc[-1])
        date = prix.index[-1].date()
        bilan["marche"] = {"instrument": ACTIF, "cloture": dernier, "seance": str(date),
                          "source": "https://finance.yahoo.com/quote/HG=F/history/"}
        if maintenant.weekday() == 2:
            bas, mediane, haut = quantiles(base, prix.tail(128), HORIZON)[-1]
            decision = signal(dernier, bas, haut, frais)
            if (maintenant.date() - date).days > 3 or dernier <= 0 or not all(
                math.isfinite(x) for x in (dernier, bas, mediane, haut)
            ) or not bas <= mediane <= haut:
                decision = "aucune position"
                motif = "Clôture trop ancienne ou prévision invalide"
            else:
                motif = "Comparaison des quantiles au prix de clôture et aux frais aller-retour"
            bilan["ticket_simulation"] = {
                "decision": decision, "motif": motif, "instrument": ACTIF,
                "prix_reference": dernier, "seance_reference": str(date),
                "horizon_seances": HORIZON, "q10": bas, "q50": mediane, "q90": haut,
                "frais_aller_retour": frais, "modele": "chronos2",
            }
    except Exception as erreur:
        bilan["erreurs"]["marche"] = str(erreur)
        if maintenant.weekday() == 2:
            bilan["ticket_simulation"] = {"decision": "aucune position", "motif": "Données de marché indisponibles"}

    sortie.mkdir(parents=True, exist_ok=True)
    temporaire = fichier.with_suffix(".tmp")
    temporaire.write_text(json.dumps(bilan, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporaire.replace(fichier)
    print(f"Collecte enregistrée : {fichier}")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--sortie", type=Path, default=Path("journal"))
    cli.add_argument("--adaptateur-indpro", type=Path)
    cli.add_argument("--frais", type=float, default=0.002)
    args = cli.parse_args()
    if not 0 <= args.frais < 1:
        cli.error("--frais doit être compris entre 0 et 1.")
    actualiser(args.sortie, args.adaptateur_indpro, args.frais)
