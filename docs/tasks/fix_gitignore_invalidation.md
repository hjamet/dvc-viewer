# Correction Invalidation (GitIgnore & Alerte Git)

## 1. Contexte & Discussion
L'utilisateur a signalé une invalidation systématique de tous ses nœuds DVC lors du passage d'un PC à un autre via GitHub. Après enquête, il s'est avéré que :
- Le `.gitignore` de `trail-rag` excluait `ragatouille/data/` à cause d'une règle `data/` trop large.
- Des fichiers non synchronisés étaient ainsi inclus dans le calcul de hash, causant des divergences entre machines.
- Un diagnostic plus précis était nécessaire dans `dvc-viewer` pour prévenir ce genre de situation.

## 2. Fichiers Concernés
- `trail-rag/.gitignore` : Correction de la règle de dossier data.
- `dvc-viewer/dvc_viewer/updater.py` : Ajout de la détection de fichiers non trackés par Git.

## 3. Objectifs (Definition of Done)
* **Correction .gitignore** : La règle `data/` devient `/data/` pour ne cibler que la racine.
* **Git Awareness** : `dvc-viewer` affiche `(NOT TRACKED BY GIT!)` à côté des fichiers problématiques dans le diagnostic.
* **Avertissement de Gloire** : Un message récapitulatif rouge prévient du risque d'invalidation entre PCs si des fichiers ne sont pas trackés.
