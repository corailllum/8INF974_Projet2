# 8INF974_Projet2
## Groupe :

**Charlotte Chanudet**

**Mahaut Galice**

**Marie Howet**

## But du projet :
Le but de ce projet est de mettre en place un Monte carlo tree search (MCTS) et un DQN pour jouée au jeux Othello sur Atari 2600. Ce README à pour but d'expliquer les différents fichier utilisée ainsi que leur . Il detaillera également les differentes étapes de lancement du projet. 

## Technologie utilisées :
- **Librairies :** gymnasium , pickle , ale_py
- **Techniques :**  Monte carlos tree search et DQN

## Organisation du Git : 
Ce GitHub est séparé en plusieurs fichier dont nous allons expliquer le rôle ici :
- **monte_carlo_vsAtari** : est un fichier python permettant de faire jouer deux MCTS l'un contre l'autre a othello
- **mcts_othello** : est le fichier python permettant de faire jouée un MCTS contre l'IA de l'atari 
- **mcts_tree.pkl** :  est une sauvegarde du modele contre l'atarie afin de pouvoir utlisée un arbre deja construit pour voir sont entrainement
- **dqn_othello** : regroupe tous le code python mis en place pour entrainer le DQN
- **dqn_othello.pth** : est une sauvegarde du modele DQN pour le lancée sur de nouvelle partie

### Prérequis
- Avoir pip d'installer (permet de suivre plus aisément les commande bash) 
- installer gymnasium


#### Lancement du MCTS contre MCTS : 
Pour lancer le fichier MCTS contre MCTS veuillez suivre les instructions suivantes :
```
python mcts_othello1.py
```

#### Lancement du MCTS contre l'Atari : 
Pour lancer le MCTS contre l'IA Atari veuillez suivre les instructions suivantes :

```
python monte_carlo_marie.py
```

#### Lancement du MCTS contre l'Atari : 
Pour lancer le DQN veuillez suivre les instructions suivantes :

```
python dqn_othello
```

Une vidéo est aussi disponible afin de voir le meilleurs modele que nous avons entrainée avec notre code (othello_demo.mp4)
