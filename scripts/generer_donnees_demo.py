"""Génère un jeu de données Vélib' + météo synthétique mais réaliste.

Contexte
--------
Le projet ClimaCity Paris s'appuie normalement sur :
  1. l'API GBFS de Vélib' Métropole (informations statiques des stations),
  2. le dépôt GitHub `lovasoa/historique-velib-opendata` (historique 15 min),
  3. les observations SYNOP de Météo-France (data.gouv.fr).

Ces trois sources sont accessibles librement sans authentification -- mais
l'environnement dans lequel nous avons développé et validé ce rendu n'a pas
d'accès sortant à Internet (sandbox d'exécution restreint). Les cellules de
téléchargement fournies dans `Spark_DIA3_Session_1.ipynb` (Section 0.1) sont
laissées **inchangées** : sur une machine avec accès Internet, elles suffisent
et ce script n'est pas nécessaire.

Ce script est notre ajout : il reconstruit, hors ligne, un jeu de données avec
exactement la même structure, les mêmes noms de colonnes et une dynamique
statistique réaliste (saisonnalité, double pic journalier, effet week-end,
effet pluie, profils de stations contrastés) afin que l'ensemble des
notebooks du projet reste exécutable de bout en bout et que les questions
métier (ruptures de stock, effet de la pluie, saisonnalité) aient une réponse
non triviale à trouver dans les données.

Échelle : par défaut, 60 stations sur 2 ans au pas horaire (~1 million de
lignes). C'est volontairement plus petit que les ~1400 stations / 12 millions
de lignes annoncées dans l'énoncé, pour que le notebook puisse être rejoué
intégralement en quelques minutes sur une machine de TP -- la volumétrie
réelle reste accessible en changeant les paramètres --n-stations et
--pas-minutes (voir --help).

Usage:
    python generer_donnees_demo.py --data-dir ../data
"""
from __future__ import annotations

import argparse
import gzip
import math
import random
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42


@dataclass
class ProfilStation:
    """Décrit le comportement type d'une station Vélib' au cours d'une journée.

    Args:
        chronotype: "residentiel" (vide le matin, se remplit le soir),
            "bureaux" (se remplit le matin, se vide le soir) ou
            "mixte" (usage homogène, faible amplitude).
        amplitude: Amplitude de la variation journalière du taux d'occupation
            (0 = constant, 1 = variation maximale).
        occupation_base: Taux d'occupation moyen "au repos" (0-1).
    """
    chronotype: str
    amplitude: float
    occupation_base: float


def generer_stations(n_stations: int, rng: np.random.Generator) -> pd.DataFrame:
    """Construit le référentiel des stations (équivalent de l'appel GBFS).

    Les stations sont réparties sur les 20 arrondissements parisiens (75001 à
    75020) avec des coordonnées jitterées autour d'un centroïde approximatif
    par arrondissement, pour obtenir un nuage de points plausible sur une
    carte Folium.

    Args:
        n_stations: Nombre de stations à générer.
        rng: Générateur aléatoire NumPy (pour la reproductibilité).

    Returns:
        DataFrame avec les colonnes station_id, name, lat, lon, capacity,
        stationCode, code_arr -- identiques à celles produites par la cellule
        de téléchargement GBFS du notebook Session 1.

    Example:
        >>> rng = np.random.default_rng(42)
        >>> df = generer_stations(10, rng)
        >>> sorted(df.columns)
        ['capacity', 'code_arr', 'lat', 'lon', 'name', 'stationCode', 'station_id']
    """
    # Centroïdes approximatifs des 20 arrondissements de Paris (lat, lon)
    centroides_arr = {
        75001: (48.8625, 2.3360), 75002: (48.8686, 2.3411),
        75003: (48.8630, 2.3610), 75004: (48.8543, 2.3573),
        75005: (48.8448, 2.3467), 75006: (48.8496, 2.3335),
        75007: (48.8567, 2.3123), 75008: (48.8718, 2.3125),
        75009: (48.8770, 2.3376), 75010: (48.8760, 2.3600),
        75011: (48.8590, 2.3800), 75012: (48.8400, 2.3900),
        75013: (48.8300, 2.3550), 75014: (48.8300, 2.3260),
        75015: (48.8420, 2.2930), 75016: (48.8620, 2.2760),
        75017: (48.8870, 2.3070), 75018: (48.8920, 2.3450),
        75019: (48.8870, 2.3820), 75020: (48.8630, 2.4010),
    }
    codes_arr = list(centroides_arr.keys())

    lignes = []
    for i in range(n_stations):
        code_arr = codes_arr[i % len(codes_arr)]
        lat0, lon0 = centroides_arr[code_arr]
        lignes.append({
            "station_id": 1000 + i,
            "name": f"Station {i:03d} -- Arr. {code_arr % 1000}",
            "lat": round(lat0 + rng.normal(0, 0.004), 6),
            "lon": round(lon0 + rng.normal(0, 0.004), 6),
            "capacity": int(rng.integers(15, 45)),
            "stationCode": f"{code_arr % 1000}{i:03d}",
            "code_arr": code_arr,
        })
    return pd.DataFrame(lignes)


def generer_profils(n_stations: int, rng: np.random.Generator) -> list[ProfilStation]:
    """Assigne un chronotype à chaque station.

    Un tiers "résidentiel", un tiers "bureaux", un tiers "mixte" -- ce qui
    garantit que le clustering K-Means du Jour 3 trouve des groupes
    réellement séparables dans le profil horaire moyen.

    Args:
        n_stations: Nombre de stations.
        rng: Générateur aléatoire NumPy.

    Returns:
        Liste de ProfilStation, une entrée par station.
    """
    chronotypes = (["residentiel"] * (n_stations // 3)
                   + ["bureaux"] * (n_stations // 3))
    chronotypes += ["mixte"] * (n_stations - len(chronotypes))
    rng.shuffle(chronotypes)

    profils = []
    for c in chronotypes:
        amplitude = rng.uniform(0.25, 0.45) if c != "mixte" else rng.uniform(0.05, 0.15)
        base = rng.uniform(0.35, 0.55)
        profils.append(ProfilStation(chronotype=c, amplitude=amplitude, occupation_base=base))
    return profils


def taux_occupation_theorique(heure: int, jour_sem: int, profil: ProfilStation) -> float:
    """Calcule le taux d'occupation moyen "sans bruit" à une heure donnée.

    Modélise un cycle journalier à deux pics (matin/soir) déphasé selon le
    chronotype de la station, et atténué le week-end (jour_sem 6=samedi,
    7=dimanche selon la convention Spark `dayofweek` où 1=dimanche).

    Args:
        heure: Heure de la journée (0-23).
        jour_sem: Jour de la semaine, convention Spark dayofweek (1=dimanche
            ... 7=samedi).
        profil: Profil comportemental de la station.

    Returns:
        Taux d'occupation théorique, borné entre 0.02 et 0.98.
    """
    est_weekend = jour_sem in (1, 7)
    # Deux pics gaussiens (matin ~8h, soir ~18h), signe inversé selon le chronotype
    pic_matin = math.exp(-((heure - 8) ** 2) / (2 * 2.2 ** 2))
    pic_soir = math.exp(-((heure - 18) ** 2) / (2 * 2.5 ** 2))

    if profil.chronotype == "residentiel":
        # Vide le matin (les gens partent en vélo -> stock baisse),
        # se remplit le soir (retour au domicile)
        signal = -pic_matin + pic_soir
    elif profil.chronotype == "bureaux":
        # Inverse : se remplit le matin (arrivée des employés), se vide le soir
        signal = pic_matin - pic_soir
    else:
        signal = 0.3 * (pic_matin + pic_soir) - 0.3

    amplitude_effective = profil.amplitude * (0.5 if est_weekend else 1.0)
    taux = profil.occupation_base + amplitude_effective * signal
    return min(max(taux, 0.02), 0.98)


def generer_meteo(dates: pd.DatetimeIndex, rng: np.random.Generator) -> pd.DataFrame:
    """Génère des observations météo horaires synthétiques (format SYNOP).

    La température suit un cycle saisonnier (sinusoïde annuelle) plus un
    cycle diurne, avec un bruit gaussien. Les épisodes de pluie sont générés
    par une chaîne de Markov à deux états (sec/pluie) pour obtenir des
    épisodes de pluie groupés dans le temps, comme dans la réalité, plutôt
    que des heures de pluie isolées et indépendantes.

    Args:
        dates: Index temporel horaire (UTC).
        rng: Générateur aléatoire NumPy.

    Returns:
        DataFrame avec les colonnes du format SYNOP utilisé par Météo-France :
        NUM_POSTE, NOM_USUEL, LAT, LON, ALTI, AAAAMMJJHH, T, U, FF, RR1, N.

    Example:
        >>> idx = pd.date_range("2022-01-01", periods=3, freq="h", tz="UTC")
        >>> df = generer_meteo(idx, np.random.default_rng(0))
        >>> list(df.columns)[:2]
        ['NUM_POSTE', 'NOM_USUEL']
    """
    n = len(dates)
    jour_annee = dates.dayofyear.values
    heure = dates.hour.values

    # Température : cycle annuel (creux en janvier, pic en juillet) + cycle diurne + bruit
    temp_saison = 11.5 - 9.5 * np.cos(2 * math.pi * (jour_annee - 15) / 365.25)
    temp_diurne = 3.0 * np.sin(2 * math.pi * (heure - 6) / 24)
    temperature_c = temp_saison + temp_diurne + rng.normal(0, 1.5, n)

    # Pluie : chaîne de Markov à 2 états pour grouper les épisodes pluvieux
    p_pluie_vers_pluie = 0.75   # persistance d'un épisode de pluie
    p_sec_vers_pluie = 0.06     # probabilité de démarrer un épisode
    etat_pluie = np.zeros(n, dtype=bool)
    en_pluie = rng.random() < 0.2
    for i in range(n):
        seuil = p_pluie_vers_pluie if en_pluie else p_sec_vers_pluie
        en_pluie = rng.random() < seuil
        etat_pluie[i] = en_pluie

    precipitation_mm = np.where(
        etat_pluie, rng.gamma(shape=1.5, scale=1.2, size=n), 0.0
    )
    humidite_pct = np.clip(
        70 + 20 * etat_pluie + rng.normal(0, 8, n), 30, 100
    )
    vitesse_vent_ms = np.clip(rng.gamma(shape=2.0, scale=2.0, size=n), 0, None)
    nebulosite = np.clip(
        (4 + 3 * etat_pluie + rng.normal(0, 1.5, n)).round(), 0, 8
    )

    df = pd.DataFrame({
        "NUM_POSTE": "75114001",
        "NOM_USUEL": "PARIS-MONTSOURIS",
        "LAT": 48.821,
        "LON": 2.337,
        "ALTI": 75.0,
        "AAAAMMJJHH": dates.strftime("%Y%m%d%H"),
        "T": (temperature_c + 273.15).round(2),   # SYNOP publie la température en Kelvin
        "U": humidite_pct.round(1),
        "FF": vitesse_vent_ms.round(2),
        "RR1": precipitation_mm.round(2),
        "N": nebulosite,
    })
    return df


def generer_disponibilite(
    stations: pd.DataFrame,
    profils: list[ProfilStation],
    dates: pd.DatetimeIndex,
    meteo_precip_mm: np.ndarray,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Génère l'historique de disponibilité Vélib' (équivalent des snapshots).

    Pour chaque couple (station, horodatage), tire un taux d'occupation autour
    de sa valeur théorique (cf. `taux_occupation_theorique`), légèrement
    relevé pendant les épisodes de pluie (moins de trajets -> plus de vélos
    qui restent en station), puis en déduit un nombre entier de vélos
    mécaniques / électriques / bornettes libres cohérent avec la capacité.

    Args:
        stations: Référentiel des stations (sortie de `generer_stations`).
        profils: Profils comportementaux, même ordre que `stations`.
        dates: Index temporel horaire commun à toutes les stations.
        meteo_precip_mm: Précipitation (mm) pour chaque horodatage de `dates`.
        rng: Générateur aléatoire NumPy.

    Returns:
        DataFrame brut au format attendu par `sc.textFile()` :
        station_id, nom_station, code_arr, capacite, velos_meca,
        velos_elec, bornettes_libres, horodatage.
    """
    n_dates = len(dates)
    heures = dates.hour.values
    jours_sem = ((dates.dayofweek.values + 1) % 7) + 1  # convention Spark dayofweek (1=dimanche)
    est_pluie = (meteo_precip_mm > 0.5)

    blocs = []
    for (_, station), profil in zip(stations.iterrows(), profils):
        taux_theo = np.array([
            taux_occupation_theorique(int(h), int(j), profil)
            for h, j in zip(heures, jours_sem)
        ])
        # Effet pluie : + 4 points d'occupation en moyenne (moins de départs)
        taux_theo = taux_theo + np.where(est_pluie, rng.normal(0.04, 0.01, n_dates), 0.0)
        taux_bruite = np.clip(taux_theo + rng.normal(0, 0.05, n_dates), 0.0, 1.0)

        capacite = int(station["capacity"])
        velos_total = np.round(taux_bruite * capacite).astype(int)
        velos_total = np.clip(velos_total, 0, capacite)
        # Répartition mécanique / électrique (environ 60% / 40%, avec bruit)
        part_elec = np.clip(rng.normal(0.4, 0.08, n_dates), 0.1, 0.7)
        velos_elec = np.round(velos_total * part_elec).astype(int)
        velos_meca = velos_total - velos_elec
        bornettes_libres = capacite - velos_total

        bloc = pd.DataFrame({
            "station_id": int(station["station_id"]),
            "nom_station": station["name"],
            "code_arr": int(station["code_arr"]),
            "capacite": capacite,
            "velos_meca": velos_meca,
            "velos_elec": velos_elec,
            "bornettes_libres": bornettes_libres,
            "horodatage": dates.strftime("%Y-%m-%dT%H:%M:%S+00:00"),
        })
        blocs.append(bloc)

    return pd.concat(blocs, ignore_index=True)


def ecrire_csv_gz_par_mois(df: pd.DataFrame, dest_dir: Path) -> None:
    """Écrit le DataFrame brut en CSV.gz mensuels (imite les releases GitHub).

    Args:
        df: DataFrame de disponibilité, doit contenir une colonne
            `horodatage` au format ISO 8601.
        dest_dir: Répertoire de destination (créé si absent).
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    mois_index = pd.to_datetime(df["horodatage"]).dt.strftime("%Y-%m")
    colonnes = ["station_id", "nom_station", "code_arr", "capacite",
                "velos_meca", "velos_elec", "bornettes_libres", "horodatage"]
    for mois, sous_df in df.groupby(mois_index):
        chemin = dest_dir / f"{mois}-01.csv.gz"
        with gzip.open(chemin, "wt", encoding="utf-8") as f:
            sous_df[colonnes].to_csv(f, sep=";", index=False)


def ecrire_parquet_partitionne(df: pd.DataFrame, dest_dir: Path) -> None:
    """Écrit le "jeu de données pré-préparé" en Parquet partitionné annee/mois.

    Reproduit ce que l'énoncé décrit comme distribué aux étudiants :
    "un extrait pré-préparé [...] au format Parquet partitionné par mois".

    Args:
        df: DataFrame de disponibilité brut (mêmes colonnes que le CSV).
        dest_dir: Répertoire racine du Parquet partitionné.
    """
    horodatage = pd.to_datetime(df["horodatage"])
    df = df.copy()
    df["annee"] = horodatage.dt.year
    df["mois"] = horodatage.dt.month
    df.to_parquet(dest_dir, partition_cols=["annee", "mois"], index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("../data"),
                         help="Racine du répertoire de données (défaut : ../data)")
    parser.add_argument("--n-stations", type=int, default=60,
                         help="Nombre de stations à simuler (défaut : 60)")
    parser.add_argument("--annee-debut", type=int, default=2022)
    parser.add_argument("--annee-fin", type=int, default=2023)
    parser.add_argument("--pas-minutes", type=int, default=60,
                         help="Résolution temporelle en minutes (défaut : 60 = horaire ; "
                              "l'historique réel est publié au pas de 15 min)")
    args = parser.parse_args()

    rng = np.random.default_rng(SEED)
    random.seed(SEED)

    data_dir = args.data_dir
    velib_raw_dir = data_dir / "velib" / "raw"
    velib_parquet_dir = data_dir / "velib" / "parquet"
    stations_csv = data_dir / "velib" / "stations_info.csv"
    meteo_csv = data_dir / "meteo" / "paris_montsouris_horaire.csv"

    print(f"[1/4] Génération de {args.n_stations} stations...")
    stations = generer_stations(args.n_stations, rng)
    profils = generer_profils(args.n_stations, rng)
    stations_csv.parent.mkdir(parents=True, exist_ok=True)
    stations.to_csv(stations_csv, index=False, sep=";")
    print(f"      -> {stations_csv}")

    debut = datetime(args.annee_debut, 1, 1, tzinfo=timezone.utc)
    fin = datetime(args.annee_fin, 12, 31, 23, 0, tzinfo=timezone.utc)
    dates = pd.date_range(debut, fin, freq=f"{args.pas_minutes}min", tz="UTC")
    print(f"[2/4] Génération de la météo horaire ({len(dates):,} points, "
          f"{debut.date()} -> {fin.date()})...")
    meteo = generer_meteo(dates, rng)
    meteo_csv.parent.mkdir(parents=True, exist_ok=True)
    meteo.to_csv(meteo_csv, index=False, sep=";")
    print(f"      -> {meteo_csv}")

    print(f"[3/4] Génération de la disponibilité Vélib' "
          f"({args.n_stations} stations x {len(dates):,} horodatages "
          f"= {args.n_stations * len(dates):,} lignes)...")
    dispo = generer_disponibilite(
        stations, profils, dates, meteo["RR1"].to_numpy(), rng
    )

    print("[4/4] Écriture des fichiers (CSV.gz mensuels + Parquet partitionné)...")
    ecrire_csv_gz_par_mois(dispo, velib_raw_dir)
    ecrire_parquet_partitionne(dispo, velib_parquet_dir)

    taille_csv = sum(f.stat().st_size for f in velib_raw_dir.glob("*.csv.gz")) / 1_048_576
    print(f"      -> {velib_raw_dir}  ({taille_csv:.1f} MB compressés)")
    print(f"      -> {velib_parquet_dir}")
    print("\nTerminé. Le jeu de données est prêt pour les notebooks du projet.")


if __name__ == "__main__":
    main()
