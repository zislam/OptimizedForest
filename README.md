# OptimizedForest

Implementation of the Optimal Subforest algorithm "OptimizedForest", which was published in:

Md Nasim Adnan and Md Zahidul Islam: Optimizing the number of trees in a decision forest to discover a subforest with high ensemble accuracy using a genetic algorithm In: Knowledge-Based Systems Vol 110, 2016 

This algorithm builds a decision forest and then works out an optimal subforest via Genetic Algorithm.

## BibTeX
```
@article{adnan2016optimizing,
  title={Optimizing the number of trees in a decision forest to discover a subforest with high ensemble accuracy using a genetic algorithm},
  author={Adnan, Md Nasim and Islam, Md Zahidul},
  journal={Knowledge-Based Systems},
  volume={110},
  pages={86--97},
  year={2016},
  publisher={Elsevier}
}
```

## Installation

Either download OptimizedForest from the Weka package manager, or download the latest release from the "Releases" section on the sidebar of Github.

## Compilation / Development

Set up a project in your IDE of choice, including weka.jar as a compile-time library.

## Valid options are:
`-S <num>;`
Seed for random number generator. (default 1)

`-I <num>`
Number of iterations for genetic algorithm. (default 20)

`-P <num>`
Initial population size for genetic algorithm. (default 20)

`-C < RandomForest | Bagging >`
Decision forest building method. (Default = RandomForest)

