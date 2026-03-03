# Hashing Portability & Git Limitations

## 1. Contexte & Discussion
L'algorithme de hash de `dvc-viewer` est conçu pour dépendre le moins possible du système d'exploitation ou de Git de l'utilisateur.

Nous avons rencontré des problèmes d'invalidation (le hash changeait d'un ordinateur à l'autre) dus à deux facteurs qui ont été corrigés :
1. **Les séparateurs de chemins (`\` vs `/`)** : Sur Windows, les chemins retournaient `\` alors que sur Linux/Mac c'était `/`. Cela est maintenant normalisé en chemins POSIX (`/`) grâce à `.as_posix()`.
2. **Les retours à la ligne (CRLF vs LF)** : Git sur Windows (via `core.autocrlf`) transforme parfois les sauts de ligne `\n` en `\r\n`. Les fichiers non-Python étaient invalidés car différents au niveau des octets. DVC-Viewer remplace désormais `\r\n` par `\n` en mémoire avant le hash.
3. **Les versions de Python (AST Stability)** : Les versions 3.12+ de Python ont introduit des changements structurels dans les nœuds AST (ex: `type_params`). L'utilisation de `ast.dump()` produisait des hashs différents selon la version de l'interpréteur. Nous utilisons désormais `ast.unparse()` qui génère un flux de code source canonique indépendant de l'implémentation interne de l'AST.
4. **Fichiers non trackés par Git (Git Awareness)** : `dvc-viewer` vérifie désormais si les fichiers qu'il hache sont suivis par Git. Si un fichier est ignoré par Git (ex: via une règle `.gitignore` trop large comme `data/` au lieu de `/data/`), il affichera un avertissement. Les fichiers non trackés sont une source majeure d'invalidation entre machines car ils ne sont pas synchronisés.

## 2. Le Piège des Dates de Modification (mtime)
L'algorithme de hash de `dvc-viewer` **ne regarde jamais** les dates de modification (mtime). Il hache uniquement le contenu et la structure du code.

**Pourquoi ?** Git ne préserve **jamais** les dates de modification originelles. Lorsqu'un fichier est récupéré via `git pull` ou `git checkout`, sa date de modification devient la date et l'heure exactes de l'exécution de la commande Git.  
Si un système externe (comme par exemple un hook `.cluster` ou un script de synchronisation distant `post_hash.py`) s'appuie sur `mtime` pour vérifier si un fichier est à jour entre deux machines synchronisées via Git, **la vérification échouera systématiquement**.

## 3. Fichiers Concernés
- `dvc_viewer/hasher.py`

## 4. Objectifs (Definition of Done)
- Assurer un hash global stable et identique pour un même état du dépôt, quel que soit l'OS (Windows, Linux, macOS).
- Documenter formellement les limites de Git vis-à-vis des dates de modification (`mtime`) afin d'éviter les bugs de synchronisation dans les hooks personnalisés des utilisateurs.
