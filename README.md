# Data Analysis Lab

Projet d’analyse de données et de prévision de séries temporelles.

## Reproduire les expériences

Python 3.12 est la version utilisée dans le workflow quotidien.

```sh
python -m pip install -r requirements.txt
python nettoyer_donnees.py
python decouper_donnees.py
python entrainer_chronos2.py --steps 100
```

Dans Colab, sélectionnez un GPU et exécutez ces commandes dans des cellules de code avec `%pip` et `!python`. Si PEFT signale un conflit avec le paquet optionnel `torchao`, exécutez `%pip uninstall -y torchao` dans cette session. Après le choix des réglages sur la validation, ajoutez `--evaluer-test` pour calculer la MAE sur les mois réservés au test. Les références sont le dernier niveau connu et, pour INDPRO, le même mois de l'année précédente. Les adaptateurs sont dans `resultats_chronos2/<série>/finetuned-ckpt/`.

## Suivi quotidien

`python suivi.py` écrit un fichier daté dans `journal/` avec les observations effectivement récupérées et les quantiles 10, 50 et 90. Les séries FRED sont INDPRO, PCOPPUSDM et CPIAUCSL (indice des prix américain, en niveau). Le flux RSS de politique monétaire de la Fed fournit le contexte. L'historique FRED peut être révisé et les dates des observations indiquent le mois mesuré. Pour évaluer les prévisions prospectives, comparez les quantiles archivés à la première publication du mois, puis calculez la MAE et la proportion de valeurs entre q10 et q90.

Le suivi utilise Chronos-2 préentraîné par défaut. Pour utiliser l'adaptateur entraîné dans Colab, décompressez-le puis lancez `python suivi.py --adaptateur-indpro /chemin/vers/INDPRO/finetuned-ckpt`. Le workflow GitHub Actions tourne à 13 h 23 UTC (8 h 23 ou 9 h 23 à Toronto) une fois fusionné sur la branche par défaut. Il peut aussi être lancé manuellement depuis Actions. Le journal écrit par ce workflow est public si le dépôt est public.

## Simulation cuivre

Le mercredi, le système compare les quantiles à 20 séances de Chronos-2 au dernier cours quotidien de HG=F et aux frais aller-retour supposés (0,20 %, réglables avec `--frais`). Il archive achat, vente ou aucune position. Le système ne transmet aucun ordre. HG=F est une série de contrats à terme, avec des effets de roulement. Utilisez l'historique du contrat réellement négocié pour étudier les exécutions réelles.

`python backtest_cuivre.py` rejoue la règle sur cinq années de données quotidiennes. Les décisions du mercredi utilisent les clôtures antérieures, les entrées simulées ont lieu à la clôture du mercredi, et les positions ne se chevauchent pas. Le CSV et le terminal donnent les résultats après frais. La baisse maximale est calculée sur les positions clôturées. Les archives quotidiennes commenceront au premier lancement réussi du workflow.
