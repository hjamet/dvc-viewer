# Optimisation du Hashing AST — Code Exécutable Only

## 1. Contexte & Discussion (Narratif)

Le hasher actuel (`dvc_viewer/hasher.py`) utilise `path.read_bytes()` pour hasher les fichiers Python. Conséquence : modifier un commentaire, un docstring, ou du whitespace invalide le pipeline DVC entier. C'est une source de friction importante.

L'utilisateur veut que seul le **code exécutable** soit pris en compte dans le hash. Deux niveaux d'optimisation ont été discutés :

- **Phase 1** : Hashing AST normalisé (ignorer commentaires, whitespace, docstrings) via `ast.dump(tree, include_attributes=False)` + strip des docstrings.
- **Phase 2** : Hashing au niveau des symboles importés (`from X import foo` → ne hasher que `foo` + sa closure transitive dans le fichier, pas `bar` qui n'est jamais importé). Utiliser le module stdlib `symtable` pour la résolution des symboles.

### Historique des décisions :
- **Approche bytecode (`compile()`) rejetée** : non reprodutible cross-version Python.
- **Analyse de flux complète rejetée** : complexité disproportionnée pour Python dynamique.
- **`ast.unparse()` seul rejeté** : n'élimine pas les docstrings.
- **`pyan3` et `libcst` rejetés** : trop lourds/orientés visualisation pour ce besoin.
- **`symtable` (stdlib) retenu** pour la Phase 2 : zéro dépendance, donne exactement les noms référencés par chaque fonction.

## 2. Fichiers Concernés
- `dvc_viewer/hasher.py` — Cœur de la modification (fonctions `compute_per_file_hashes()` et `compute_aggregate_hash()`)
- `tests/test_invalidation_diagnostic.py` — Tests existants à enrichir

## 3. Objectifs (Definition of Done)

### Phase 1 — AST normalisé
* Les commentaires, whitespace et docstrings dans les `.py` ne changent plus le hash
* Les fichiers non-Python (`.yaml`, `.json`, `.sh`) gardent le hash `read_bytes()` classique
* Les tests existants passent toujours + nouveaux tests de non-invalidation

### Phase 2 — Symbol-level hashing
* `from X import foo` → seul `foo` + ses dépendances internes sont hashés
* `import X` ou `from X import *` → fallback au hash complet du fichier (conservateur)
* Les constantes/globals utilisés par les fonctions importées sont inclus dans le hash
* Le module `symtable` (stdlib) est utilisé pour la résolution des symboles
