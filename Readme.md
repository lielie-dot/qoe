# Prédiction du QoE

Développement d'un outil de prédiction de la satisfaction cliente à partir de données de type réseau.

## Table des matières

<!-- TOC -->
* [Prédiction du QoE](#prédiction du QoE)
  * [Table des matières](#table-des-matières)
  * [Éditeurs](#éditeurs)
  * [Fonctionnalités](#fonctionnalités)
  * [Prérequis](#prérequis)
  * [Installation](#installation)
  
<!-- TOC -->


## Éditeurs
- [@Ndeye-Emilie Mbengue](Ndeye.Mbengue@telecomnancy.eu)
## Fonctionnalités

- Prédiction d'une satisfaction cliente à partir de 4 mesures
- Visualisation des étapes clefs de segmentation et choix de variables explicatives
- Test du modèle avec un fichier externe (.csv)

## Prérequis

**Python3.10:**

https://www.python.org/downloads/release/python-3100/

## Installation

L'ensemble du code de ce projet se trouve dans le répertoire code.
Toutes les commandes et directives citées ultérieurement seront exécutées à partir de celui-ci.


###  Configuration d'environnement virtuel


Pour installer notre projet sans interférer avec les packages déjà installés, il est conseillé d'utiliser un environnement virtuel :
 1) Ouvrez un terminal ou une invite de commande.
 2) Naviguez vers le dossier racine de votre projet.
 3) Créez un nouvel environnement virtuel en exécutant la commande suivante :
    ```bash
    python3 -m venv env
    ```
 4) Activez l'environnement virtuel en exécutant la commande suivante :
    
    Sur Linux/Mac :  
    ```bash
    source env/bin/activate
    ```
  
 5) Vous pouvez maintenant installer les packages de notre projet dans cet environnement virtuel à l'aide de pip.
 ### Installation des packages nécessaires
Pour installer les packages nécessaires pour exécuter notre projet, il vous faut exécuter la ligne de commande suivante sur votre terminal : 
  ```bash
  pip install requirements.txt -r .
  ```

## Usage/Examples

Pour utiliser l'interface graphique il faut taper dans la ligne de commande du terminal les commandes suivantes:

 ``` bash
export FLASK_APP=src/app.py
flask run

 ```
