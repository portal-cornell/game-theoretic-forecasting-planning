## Game Theoretic Forecasting and Planning

This is an official implementation for the IROS 2023 paper:\
**A Game Theoretic Framework for Joint Forecasting and Planning**
<br>
<a href="https://kushal2000.github.io/">Kushal Kedia</a>,
Prithwish Dan,
<a href="https://www.sanjibanchoudhury.com/">Sanjiban Choudhury</a>

<p align="center">
  <img src="docs/framework.png">
</p>

We implement a game-theoretic framework for joint forecasting and planning on the [CrowdNav](https://github.com/vita-epfl/CrowdNav) environment, and compare with the industry standard of Maximum-Likelihood Estimation (MLE) based forecasting and planning.

### Setup

Setup environments following the [SETUP.md](docs/SETUP.md)

### Training

Train MLE-Forecaster and Nominal Planner
```
cd crowd_nav
python train_mle_forecaster.py
python train_nom_planner.py
```

Finetune the above models using the game-theoretic framework.
```
python train_forecaster_planner_game.py
```


### Evaluation
The following code compares the costs and collision rates of our approach with MLE.
```
cd crowd_nav
python evaluate.py
```

### Results
<table border="0">
 <tr align="center">
    <td><img src="docs/SAFE.gif" alt>
    <em>Safe Planning with Adversarial Forecasts</em></td>
    <td><img src="docs/MLE.gif" alt>
    <em>Collision with MLE Forecasting and Planning</em></td>
 </tr>
</table>

### Work in Progress
We will soon release the implementation of our algorithm on the ETH-UCY benchmark.

### Acknowledgement

This repository borrows code from [Social-NCE](https://github.com/vita-epfl/social-nce/).

### Citation

```bibtex
@inproceedings{kedia2023game,
  title={A Game-Theoretic Framework for Joint Forecasting and Planning},
  author={Kedia, Kushal and Dan, Prithwish and Choudhury, Sanjiban},
  booktitle={IROS},
  year={2023}
}
```
