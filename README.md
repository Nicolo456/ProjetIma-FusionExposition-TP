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
