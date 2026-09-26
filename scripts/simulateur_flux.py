"""Simulateur de flux temps réel Vélib' pour Structured Streaming (Jour 2, après-midi).

Contexte
--------
L'énoncé du projet (section 3.1, Source 3) mentionne "un micro-service de
simulation [...] fourni aux étudiants" qui rejoue en accéléré les données
historiques et écrit des fichiers JSON dans un répertoire surveillé par
Spark. Ce micro-service ne figurait pas parmi les fichiers du cours qui nous
ont été distribués : ce script est donc **notre implémentation** de ce rôle,
conçue pour reproduire fidèlement le contrat attendu par
`Spark_DIA3_Session_4.ipynb` (Section 2.2 et schéma `schema_flux`) :

- un fichier JSON par micro-batch, contenant une liste d'observations,
- une observation par station, avec exactement les colonnes de
  `schema_flux` (station_id, nom_station, code_arr, capacite, velos_meca,
  velos_elec, bornettes_libres, horodatage),
- des horodatages ancrés sur l'heure "réelle" (UTC, `datetime.now`) plutôt
  que sur des dates historiques 2022-2023, pour que les mécanismes de
  watermark et de données tardives du notebook (qui comparent l'horodatage
  des événements à l'heure de traitement) restent démonstratifs.

Nous réutilisons la dynamique comportementale (chronotypes, effet pluie)
définie dans `generer_donnees_demo.py` pour que le flux simulé ait la même
cohérence statistique que l'historique batch.

Usage:
    python simulateur_flux.py --output ../data/output/stream_input --vitesse 3

    # Arrêt automatique après 20 fichiers (pratique pour les tests / la
    # correction, plutôt que de laisser tourner indéfiniment) :
    python simulateur_flux.py --output ../data/output/stream_input --max-fichiers 20
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from generer_donnees_demo import (
    SEED,
    generer_profils,
    generer_stations,
    taux_occupation_theorique,
)


def tirer_snapshot(stations, profils, instant: datetime, en_pluie: bool, rng) -> list[dict]:
    """Tire un snapshot instantané pour toutes les stations.

    Args:
        stations: DataFrame des stations (sortie de `generer_stations`).
        profils: Liste de ProfilStation, même ordre que `stations`.
        instant: Horodatage UTC à associer aux observations.
        en_pluie: Si True, applique le décalage "pluie" au taux d'occupation.
        rng: Générateur aléatoire NumPy.

    Returns:
        Liste de dictionnaires, un par station, prêts à être sérialisés en
        JSON avec les clés attendues par `schema_flux`.
    """
    jour_sem = ((instant.weekday() + 1) % 7) + 1  # convention Spark dayofweek
    lignes = []
    for (_, station), profil in zip(stations.iterrows(), profils):
        taux = taux_occupation_theorique(instant.hour, jour_sem, profil)
        if en_pluie:
            taux = min(taux + rng.normal(0.04, 0.01), 0.98)
        taux = float(np.clip(taux + rng.normal(0, 0.05), 0.0, 1.0))

        capacite = int(station["capacity"])
        velos_total = int(round(taux * capacite))
        velos_total = max(0, min(velos_total, capacite))
        part_elec = float(np.clip(rng.normal(0.4, 0.08), 0.1, 0.7))
        velos_elec = int(round(velos_total * part_elec))
        velos_meca = velos_total - velos_elec
        bornettes_libres = capacite - velos_total

        lignes.append({
            "station_id": int(station["station_id"]),
            "nom_station": station["name"],
            "code_arr": int(station["code_arr"]),
            "capacite": capacite,
            "velos_meca": velos_meca,
            "velos_elec": velos_elec,
            "bornettes_libres": bornettes_libres,
            # Format ISO 8601 sans microsecondes -- lisible par
            # `to_timestamp()` côté Spark une fois le schéma explicite fourni.
            "horodatage": instant.strftime("%Y-%m-%dT%H:%M:%S+00:00"),
        })
    return lignes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True,
                         help="Répertoire surveillé par Spark (readStream)")
    parser.add_argument("--n-stations", type=int, default=60,
                         help="Nombre de stations simulées (doit être cohérent "
                              "avec le batch si vous comparez les deux -- défaut : 60)")
    parser.add_argument("--vitesse", type=float, default=3.0,
                         help="Facteur d'accélération (indicatif, affiché en log)")
    parser.add_argument("--intervalle-sec", type=float, default=5.0,
                         help="Secondes réelles entre deux micro-batchs écrits")
    parser.add_argument("--max-fichiers", type=int, default=0,
                         help="Nombre maximal de fichiers à écrire (0 = illimité, "
                              "s'arrête uniquement sur Ctrl+C)")
    parser.add_argument("--proba-pluie", type=float, default=0.0,
                         help="Probabilité qu'un micro-batch soit marqué 'pluie' "
                              "(0.0-1.0) -- utile pour varier la démo")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    stations = generer_stations(args.n_stations, rng)
    profils = generer_profils(args.n_stations, rng)

    print(f"[simulateur_flux] {args.n_stations} stations -- écriture toutes les "
          f"{args.intervalle_sec:.1f}s dans {args.output} (vitesse x{args.vitesse:.0f})")

    i = 0
    try:
        while args.max_fichiers == 0 or i < args.max_fichiers:
            instant = datetime.now(timezone.utc)
            en_pluie = bool(rng.random() < args.proba_pluie)
            snapshot = tirer_snapshot(stations, profils, instant, en_pluie, rng)

            chemin = args.output / f"snapshot_{instant.strftime('%Y%m%dT%H%M%S')}_{i:05d}.json"
            chemin_tmp = chemin.with_suffix(".json.tmp")
            # Écriture atomique : on écrit dans un fichier temporaire puis on
            # renomme, pour que Spark ne lise jamais un fichier à moitié écrit.
            with open(chemin_tmp, "w", encoding="utf-8") as f:
                json.dump(snapshot, f)
            chemin_tmp.rename(chemin)

            i += 1
            print(f"  [{i}] {chemin.name}  ({len(snapshot)} stations, "
                  f"pluie={en_pluie})")
            time.sleep(args.intervalle_sec)
    except KeyboardInterrupt:
        print("\n[simulateur_flux] Arrêt demandé (Ctrl+C).")

    print(f"[simulateur_flux] Terminé -- {i} fichiers écrits dans {args.output}")


if __name__ == "__main__":
    main()
