# Résolution de la corruption du RWLock DVC

## 1. Contexte & Discussion (Narratif)
> *L'utilisateur signalait des erreurs fréquentes de corruption du fichier `.dvc/tmp/rwlock` lors de l'utilisation de dvc-viewer.*

L'analyse a révélé que `dvc-viewer` déclenchait indirectement cette corruption. En appelant `dvc status --json` pour détecter si un run était en cours, `dvc-viewer` forçait DVC à acquérir un read-lock. Or, le mécanisme interne de DVC (`rwlock.py`) réécrit systématiquement le fichier JSON à la sortie du lock, créant une contention avec un éventuel `dvc repro` tournant en parallèle.

La solution a consisté à rendre la détection "passive" : on lit le fichier JSON directement sans passer par DVC, et on n'appelle DVC en fallback que si on est sûr qu'aucun run n'est en cours.

## 2. Fichiers Concernés
- `dvc_viewer/parser.py` : Inversion de la logique de détection et nouvelle fonction `_safe_read_rwlock`.
- `dvc_viewer/server.py` : Utilisation de la lecture sécurisée lors de l'arrêt du pipeline.
- `tests/test_parser.py` : Nouveaux tests unitaires pour la robustesse du RWLock.

## 3. Objectifs (Definition of Done)
* **Élimination de la contention** : `dvc status` n'est plus appelé quand un run est détecté via le rwlock.
* **Robustesse aux corruptions** : `_safe_read_rwlock` gère les fichiers vides/partiels par un retry et un nettoyage automatique des fichiers orphelins.
* **Tests validés** : 100% de succès sur les tests de lecture/nettoyage du rwlock.
