# 8INF974_Projet2 : Mise en place d'un MCTS et d'un DQN pour jouée aux jeux atari
## Groupe :

**Charlotte Chanudet**

**Mahaut Galice**

**Marie Howet**

## But du projet :
Le but de ce projet est de mettre en place un Monte carlo tree search (MCTS) et un DQN pour jouée à des jeux sur Atari 2600.Cela est possible au travere de la bibliotheque Gymnasium. Ce README à pour but d'expliquer les différents fichier utilisée. Il detaillera également les differentes étapes de lancement du projet. 

## Technologie utilisées :
- **Librairies :** gymnasium , pickle , ale_py
- **Techniques :**  Monte carlos tree search et DQN
- **Jeux :** Othello, Pong

## Organisation du Git : 
Ce GitHub est séparé en deux dossier distinct en fonction des jeux choisi :
- ***Pong*** est lui dediée au jeux du meme nom et regroupe les fichier suivant :
    - **DQN_Pong_Entrainement.py et DQN_Pong_Entraine.pth** : correspond respectivement a l'entrainement du DQN sur 500 partie et la sauvergarde du modele.
    - **DQN_Pong_Rentrainement.py et DQN_Pong_Rentraine.pth** : correspond respectivement a l'entrainement du DQN sur 1000 partie et la sauvergarde du modele.
    - **mcts_pong.py** : est la code python mis en place pour faire jouée un MCTS contre l'IA de PONG
    -**Pong_Demo.mp4 et Pong_Demo_Rentraine.mp4** : sont des videos demo de nos entrainement effectuée avec les dqn.


- ***Othello*** regroupe les code mis en place pour le jeux othello et groupe les fichier suivant :
    - **monte_carlo_vsAtari** : est un fichier python permettant de faire jouer deux MCTS l'un contre l'autre a othello
    - **mcts_othello** : est le fichier python permettant de faire jouée un MCTS contre l'IA de l'atari 
    - **mcts_tree.pkl** :  est une sauvegarde du modele contre l'atarie afin de pouvoir utlisée un arbre deja construit pour voir sont entrainement
    - **dqn_othello** : regroupe tout le code python mis en place pour entrainer le DQN
    - **dqn_othello.pth** : est une sauvegarde du modele DQN pour le lancée sur de nouvelle partie





### Prérequis
- Avoir pip d'installer (permet de suivre plus aisément les commande bash) 
- installer gymnasium


#### Pong

**Lancement du MCTS  :**
Pour lancer le MCTS contre l'IA Atari veuillez suivre les instructions suivantes :

```
python mcts_pong.py
```

**Lancement de l'entrainement du DQN :**
Pour lancer le DQN veuillez suivre les instructions suivantes :

```
python DQN_Pong_Entrainement.py
```

**Lancement du reentrainement du DQN :**

```
python DQN_Pong_Rentrainement.py
```
Pour les deux entrainement effectuée avec les DQNs, des vidéos sont disponible(Pong_Demo.mp4 et Pong_Demo_Rentraine.mp4). 


#### Othello
**Lancement du MCTS contre MCTS :**
Pour lancer le fichier MCTS contre MCTS veuillez suivre les instructions suivantes :

```
python MCTS_VS_MCTS.py
```

**Lancement du MCTS contre l'Atari :**
Pour lancer le MCTS contre l'IA Atari veuillez suivre les instructions suivantes :

```
python MCTS_VS_Atari.py
```

**Lancement du DQN :**
Pour lancer le DQN veuillez suivre les instructions suivantes :

```
python dqn_othello.py
```

Une vidéo est aussi disponible afin de voir le meilleurs modele que nous avons entrainée avec notre code (othello_demo.mp4)


