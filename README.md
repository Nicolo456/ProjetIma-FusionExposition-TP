# README.md

## Description du projet

Ce projet est un code permettant d'effectuer une fusion d'exposition, tel que présenté dans le papier "Mertens, Tom & Kautz, Jan & Van Reeth, Frank. (2007). Exposure Fusion. Pacific Graphics. 382 - 390. 10.1109/PG.2007.17. "
Il a été développé dans le cadre du cours CSC_4IM01_TP de la filière IMA de Télécom Paris par Damien Pelissier et Nicodème Gorge, supervisés par Saïd Ladjal.

## Etapes d'installation

1. Cloner le dépôt GitHub `git clone https://github.com/Nicolo456/ProjetIma-FusionExposition-TP.git`.
2. Deplacer vous dans le dossier du code `cd code`.
3. Créer un environnement virtuel `python -m venv env`.
4. Activer l'environnement virtuel `source env/bin/activate`.
5. Installer les dépendances nécessaires `pip install -r requirements.txt`.
6. Exécuter le code `python AMODIFIERSAID.py`. Pour plus de précision utiliser l'aide avec `python AMODIFIERSAID.py --help`. Le fichier est minimal et chaque modification est expliqué.
7. Pour voir les différents filtres colorimétriques, lancer la commande avec le flag -e (`python3 AMODIFIERSAID.py -e`). Pour éviter de recalculer les images vous pouver ajouter le flag -r (`python3 AMODIFIERSAID.py -e -r`).

## Description

Nous avons ajouté une description en haut de chaque fichier pour expliquer ce qu'il contient.

### Abréviation

Petit dictionnaire des abréviations utilisées dans le code :

- Dès qu'il y a un 's' à la fin c'est qu'on parle d'un vecteur de l'objet de base
  _Exemple_ : `img` pour une image, `imgs` pour une liste d'images, `imgss` pour une liste de liste d'images.
- `wm` pour weight map
- `n_` pour dire que l'objet est normalisé
  _Exemple_ : `n_img` pour une image normalisée
- `sorted_` pour dire que l'objet est trié. Dans le cas des pyramides gaussiennes cela signifie que chaque element de la liste est une liste contenant le même niveau des différentes pyramides. => Utiliser la fonction `inspect_list_structure` pour visualiser (dans le package `fusion_exposition.inspect_list_structure`)
