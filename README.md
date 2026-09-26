# ClimaCity Paris — Analyse de données de mobilité urbaine avec Apache Spark et PySpark

**Module** : Traitement de données massives — Apache Spark / PySpark
**Projet réalisé par** : Nermine Khadhraoui & Imen Khlifi (binôme)

---

## 1. Ce que nous avons compris du projet

ClimaCity Paris nous met dans la peau d'un binôme de data engineers chargé de construire, sur trois journées, une plateforme de données pour un opérateur de mobilité parisien. L'objectif pédagogique n'est pas de produire un système de production, mais de traverser l'ensemble du spectre Spark sur un cas d'usage cohérent : RDD le premier matin, DataFrame l'après-midi, Spark SQL et Delta Lake le lendemain matin, Structured Streaming l'après-midi, puis MLlib (régression, clustering) et optimisation le troisième jour. Chaque brique s'appuie sur la précédente — la table consolidée du Jour 1 alimente le SQL et le streaming du Jour 2, qui alimentent à leur tour les features du modèle de prédiction du Jour 3.

Le point de départ n'était pas une page blanche : on nous a fourni six notebooks (`Spark_DIA3_Session_1` à `_6`) déjà structurés, avec les explications pédagogiques, une partie du code déjà écrite, et des trous marqués `[EXERCICE]` / `[Réponse]` / `TODO` à compléter. Notre travail a consisté à :

1. comprendre l'intention de chaque exercice à partir du contexte (cellules Markdown, code environnant, cellules suivantes qui réutilisent nos variables) ;
2. écrire le code manquant en respectant l'API et le style déjà en place dans le notebook ;
3. documenter chaque ajout, à la fois sur ce qu'il fait et sur pourquoi cette façon de faire est pertinente en Spark (plutôt qu'un simple équivalent Pandas) ;
4. vérifier que l'ensemble s'exécute réellement, de bout en bout, sur des données cohérentes.

Le notebook Session 6 (optimisation et bilan) nous a été fourni **entièrement fonctionnel**, sans aucune cellule à compléter — nous l'avons revérifié et laissé tel quel.

---

## 2. Composition du rendu

```
ClimaCity_Paris_PySpark/
├── README.md                          <- ce fichier
├── notebooks/
│   ├── Spark_DIA3_Session_1.ipynb     <- Jour 1 matin  : API RDD
│   ├── Spark_DIA3_Session_2.ipynb     <- Jour 1 après-midi : API DataFrame
│   ├── Spark_DIA3_Session_3.ipynb     <- Jour 2 matin  : Spark SQL + Delta Lake
│   ├── Spark_DIA3_Session_4.ipynb     <- Jour 2 après-midi : Structured Streaming
│   ├── Spark_DIA3_Session_5.ipynb     <- Jour 3 matin  : MLlib, K-Means, MLflow
│   └── Spark_DIA3_Session_6.ipynb     <- Jour 3 après-midi : optimisation, bilan (fourni complet)
├── scripts/
│   ├── generer_donnees_demo.py        <- générateur de données de secours (notre ajout, voir §4)
│   └── simulateur_flux.py             <- simulateur de flux temps réel (notre ajout, voir §4)
└── data/                              <- données Vélib' + météo (générées, voir §4)
    ├── velib/{raw, parquet, stations_info.csv}
    ├── meteo/paris_montsouris_horaire.csv
    └── output/                        <- créé et rempli par les notebooks à l'exécution
```

**Important — continuité de noyau (kernel) entre notebooks.** Comme dans le déroulé original du cours, certains notebooks ne sont pas autonomes : ils réutilisent des variables (`spark`, `sc`, `step3`, `df`, des sessions de streaming...) définies dans le notebook précédent du même jour. Il faut donc les exécuter **dans le même noyau Jupyter, dans l'ordre, sans redémarrer** :

- `Session_1` → `Session_2` (Jour 1, un seul noyau)
- `Session_3` → `Session_4` (Jour 2, un seul noyau)
- `Session_5` → `Session_6` (Jour 3, un seul noyau)

C'est volontaire (voir par exemple `Session_2` qui se termine par `spark.stop()` — il n'y a pas de nouvel appel `SparkSession.builder` dans cette cellule car la session vient de `Session_1`).

---

## 3. Couverture technique

| Journée | Module Spark | Ce qui a été complété par nous |
|---|---|---|
| J1 matin | RDD | `SparkSession.builder`, `textFile`, `map`/`filter`/`flatMap`, `parse_ligne`, pipeline `step1→step2→step3`, `reduceByKey` vs `groupByKey`, `join` RDD |
| J1 après-midi | DataFrame | lecture Parquet, `describe`, détection de nulls/anomalies, pipeline de nettoyage en 7 étapes, colonne `statut`, chargement météo SYNOP, jointure `broadcast`, écriture Parquet partitionnée |
| J2 matin | Spark SQL / Delta | 3 questions métier en SQL (ruptures, effet pluie, saisonnalité), fenêtrage `LAG`/`LEAD`/`ROW_NUMBER`/moyenne mobile, écriture Delta, `append`, time-travel, `MERGE INTO` |
| J2 après-midi | Structured Streaming | `readStream` avec schéma explicite, sink console, fenêtres glissantes + `withWatermark`, sink Delta, alertes via `foreachBatch` avec état inter-batchs, injection de données tardives |
| J3 matin | MLlib | features cycliques (sin/cos), features de lag, split temporel train/test, `Pipeline` (`VectorAssembler`→`StandardScaler`→`GBTRegressor`), `K-Means` + méthode du coude, carte Folium, `CrossValidator` (déjà fourni), MLflow (déjà fourni) |
| J3 après-midi | Optimisation | fourni entièrement fonctionnel — relu et revérifié, non modifié |

---

## 4. Choix techniques importants

### 4.1 Données synthétiques de secours (`scripts/generer_donnees_demo.py`)

L'énoncé s'appuie sur trois sources réelles : l'API GBFS de Vélib' Métropole, l'historique GitHub `lovasoa/historique-velib-opendata`, et les observations SYNOP de Météo-France. Les cellules de téléchargement fournies dans `Session_1` (section 0.1) sont **inchangées** et fonctionnent sur une machine avec un accès Internet normal.

Nous avons développé et vérifié ce rendu dans un environnement d'exécution sans accès sortant à Internet (sandbox restreint par une liste blanche de domaines). Pour ne pas bloquer sur cette contrainte d'environnement, nous avons écrit `generer_donnees_demo.py` : un générateur qui reproduit **exactement le même schéma** que les données réelles (mêmes colonnes, mêmes types, mêmes fichiers de sortie — CSV.gz mensuels, `stations_info.csv`, Parquet partitionné par année/mois), mais avec une dynamique statistique construite pour rester pédagogiquement intéressante :

- un cycle jour/nuit à deux pics (matin/soir), déphasé selon trois chronotypes de stations (résidentiel / bureaux / mixte), pour que le clustering K-Means du Jour 3 ait un sens réel (nous l'avons vérifié : nos tests retrouvent bien 3 clusters distincts correspondant aux 3 chronotypes injectés) ;
- des épisodes de pluie groupés dans le temps (chaîne de Markov à deux états, pas des heures de pluie isolées) ;
- un effet pluie mesurable sur le taux d'occupation (+4 points environ), pour que la Question 2 du Jour 2 (« la pluie réduit-elle le taux d'occupation ? ») ait une réponse non triviale à trouver — confirmé par notre test d'intégration : le taux d'occupation moyen est systématiquement plus élevé sous la pluie, dans les 20 arrondissements.

`Session_1` appelle ce script automatiquement (cellule ajoutée juste après les cellules de téléchargement réel, clairement commentée) **uniquement si les fichiers réels sont absents** — c'est un filet de sécurité, pas un remplacement. Il est paramétrable (`--n-stations`, `--annee-debut/fin`, `--pas-minutes`) : nous avons utilisé 60 stations sur 2022-2023 au pas horaire (≈1,05 million de lignes) pour rester rejouable rapidement ; l'échelle réelle de l'énoncé (~1400 stations, pas de 15 min, ~12 millions de lignes) reste accessible en changeant ces paramètres si vous disposez d'une machine plus puissante ou de plus de temps.

### 4.2 Simulateur de flux temps réel (`scripts/simulateur_flux.py`)

L'énoncé mentionne un micro-service de simulation « fourni aux étudiants » pour la partie Structured Streaming — il ne figurait pas parmi les fichiers reçus. Nous l'avons donc écrit nous-mêmes, en respectant le contrat attendu par `schema_flux` dans `Session_4` (mêmes colonnes, écriture JSON atomique via fichier temporaire + renommage). Contrairement à l'énoncé qui demande de le lancer « à la main » dans un terminal séparé, nous le lançons automatiquement en sous-processus depuis le notebook (cellule ajoutée, clairement commentée), pour que l'ensemble reste rejouable en un seul « Run All » — conformément à la contrainte de reproductibilité du projet.

### 4.3 Durées de streaming réduites (`Session_4`)

L'énoncé original prévoit des pauses de 20 à 60 secondes pour laisser le temps à plusieurs micro-batchs de s'exécuter sur un flux « temps réel ». Nous avons resserré ces durées (triggers de 2 à 5 s, pauses de 6 à 25 s au lieu de 20-60 s) pour que la partie streaming reste rejouable en quelques minutes lors de la correction, sans changer la logique pédagogique — le nombre de micro-batchs observés reste suffisant pour illustrer fenêtres glissantes, sink Delta, alertes `foreachBatch` et watermark. C'est documenté dans une cellule Markdown ajoutée à cet effet dans le notebook.

### 4.4 Correction d'une incohérence de schéma entre Session 1 et Session 2

En testant l'exécution réelle de `Session_1` puis `Session_2` à la suite, nous avons découvert que notre implémentation de l'Exercice 8 (Session 1) enrichit chaque enregistrement d'un champ `taux_occupation` calculé — nécessaire pour l'Exercice 13 (flatMap) plus loin dans le même notebook. Or la cellule 1 de `Session_2` (fournie, non un exercice) construit un DataFrame à partir de ce même RDD avec un schéma `schema_velib` à 8 colonnes, sans `taux_occupation` : l'appel `spark.createDataFrame(row_rdd, schema=schema_velib)` échouait avec `LENGTH_SHOULD_BE_THE_SAME` (9 champs pour 8 attendus). Nous avons ajouté le champ manquant au schéma plutôt que de retirer une information utile à `step3` — la correction et sa justification figurent en commentaire directement dans la cellule concernée.

### 4.5 Delta Lake

Toutes les cellules Delta Lake (écriture, `append`, time-travel, `MERGE INTO`) suivent l'API officielle Delta Lake 3.x (`DeltaTable.forPath`, `.merge().whenMatchedUpdateAll().whenNotMatchedInsertAll().execute()`, `option("versionAsOf", n)`). Nous les avons exécutées avec succès sur notre environnement de travail final (conda + `delta-spark`, accès réseau standard pour le téléchargement du JAR `io.delta` depuis Maven Central au premier lancement de la session Spark) : écriture initiale, ajout `append`, lecture de versions passées et `MERGE INTO` produisent les résultats attendus.

### 4.6 Vérification et exécution

Nous avons vérifié le projet en deux temps. D'abord des tests ciblés pendant le développement (composants isolés puis intégration réelle de `Session_1`+`Session_2` sur un sous-échantillon de nos données), qui nous ont permis de repérer des problèmes avant de les corriger — notamment le bug de schéma du §4.4, et l'ajustement des durées de streaming du §4.3. Puis une exécution complète, **notebook par notebook, de bout en bout**, sur notre environnement de travail définitif (conda, accès Internet complet) : les six notebooks s'exécutent sans erreur dans l'ordre indiqué au §2, y compris les sections Delta Lake et le téléchargement/génération des données. Le clustering K-Means retrouve des groupes de stations cohérents avec leurs profils d'usage, l'effet de la pluie sur le taux d'occupation est visible dans les résultats SQL du Jour 2, et le modèle `GBTRegressor` obtient un RMSE nettement inférieur à l'écart-type brut de la variable cible, confirmant qu'il apprend un signal réel.

---

## 5. Transparence sur l'usage de l'IA

Conformément à la consigne (« il n'est pas interdit d'avoir recours à des agents [...] je vous demanderai d'être transparent »), nous précisons ici notre usage de l'IA : nous nous sommes fait assister par **Claude (Anthropic)** comme outil d'aide au codage pour une partie de ce projet, sous notre supervision — c'est nous qui avons fixé le périmètre, demandé les vérifications, relu chaque cellule et validé (ou corrigé) les résultats avant de les considérer comme acquis.

Usages principaux :

- **Cellules `[EXERCICE]`/`[Réponse]`/`TODO` des six notebooks** : premier jet du code manquant, à partir de notre lecture du contexte pédagogique (cellules Markdown, code environnant, variables réutilisées plus loin). Relu et vérifié par exécution réelle sur nos données pour chacun des six notebooks.
- **`scripts/generer_donnees_demo.py` et `scripts/simulateur_flux.py`** : ces deux fichiers, mentionnés dans l'énoncé mais absents de ce que nous avions reçu, ont été écrits avec l'aide de Claude à partir du schéma de données attendu par les notebooks (colonnes, types, format de sortie). Nous avons vérifié que les données produites ont une dynamique réaliste et pédagogiquement utile (cycle jour/nuit, effet pluie, profils de stations contrastés pour le clustering, cf. §4.1).
- **Correction du bug de schéma (§4.4)** : détecté en exécutant réellement `Session_1` puis `Session_2` à la suite (incompatibilité entre le dictionnaire produit par notre implémentation de l'Exercice 8 et le schéma `schema_velib` fourni dans `Session_2`). Diagnostic et correctif proposés par Claude, vérifiés par nous en ré-exécutant les deux notebooks jusqu'au bout.
- **Ajustement des durées de streaming (§4.3)** : les pauses de 20 à 60 secondes de l'énoncé original ont été resserrées pour que `Session_4` reste rejouable rapidement, sans perdre le nombre de micro-batchs nécessaire pour observer fenêtres glissantes, alertes et watermark — choix que nous avons validé en réexécutant le notebook et en vérifiant que les résultats attendus (fenêtres fermées, alertes détectées) sont bien produits.

Le raisonnement Spark derrière chaque exercice (pourquoi `reduceByKey` plutôt que `groupByKey`, pourquoi `broadcast` sur la table météo, pourquoi un split temporel plutôt qu'aléatoire pour le ML, etc.) est celui du cours ; nous l'avons vérifié cellule par cellule et sommes en mesure de l'expliquer.

---

## 6. Reproduire l'exécution

1. Créer l'environnement conda fourni par le cours (`pyspark>=3.5`, `delta-spark`, `mlflow`, `folium`, `plotly`, `openmeteo-requests`, `jupyterlab`, `pandas`, `numpy`).
2. Placer ce dossier tel quel (les chemins dans les notebooks sont relatifs : `Path("../data")` depuis `notebooks/`).
3. Ouvrir `notebooks/Spark_DIA3_Session_1.ipynb` dans JupyterLab et exécuter toutes les cellules (« Run All »). Si vous disposez d'un accès Internet, les cellules de téléchargement réel (section 0.1) récupèrent les vraies données Vélib'/GBFS ; sinon, notre filet de sécurité (§4.1) génère un jeu de données synthétique équivalent.
4. Dans le **même noyau**, ouvrir et exécuter `Spark_DIA3_Session_2.ipynb`.
5. Redémarrer un noyau, exécuter `Spark_DIA3_Session_3.ipynb` puis, dans le même noyau, `Spark_DIA3_Session_4.ipynb`.
6. Redémarrer un noyau, exécuter `Spark_DIA3_Session_5.ipynb` puis, dans le même noyau, `Spark_DIA3_Session_6.ipynb`.
7. Le Spark UI est disponible sur `http://localhost:4040` pendant l'exécution de chaque notebook ; MLflow UI via `mlflow ui --backend-store-uri data/output/mlruns --port 5000` après `Session_5`.

---

## 7. Dépôt du code

Ce README peut être déposé seul sur Hetic Learn, accompagné du lien vers ce dépôt GitHub contenant le code complet (notebooks, scripts, données). Une fois le dépôt créé et poussé (voir les commandes fournies avec la livraison), remplacer la ligne « Dépôt du code » en haut de ce fichier par son URL avant le dépôt final sur Hetic Learn.
