# Framework for Artificial Evolution

## ℹ️ Overview

The Framework for Artificial Evolution (FAE) is a basic implementation of evolutionary computation and genetic programming used to run experiments under various models.

## ⚙️ Standard Evolution Methods

Methods contained withing the evolve.py are standard methods used for evolution are usually operate independent of the topic of evolution.

## 📚 Models

Individual modules used to implement new forms of evolution. Broken into several files:

- main: File used to run experiment
- methods: General evolution methods such as mutation and recombination
- model: Class based implementation
- plot: Specific methods of graphic results or models

Models can be added and removed without impacting any other part of the FAE.

### Self-Modifying Linear Genetic Programming (smlgp)
Advanced model using memory banks and a more structured assembly like language to evolve systems capable of self modification.

### Directed Acyclic Graph Genetic Programming (daggp)
A more classic methods of GP which is also used to evolve standard trees.

### Turing Machine Genetic Programming (tmgp)
A simple model used to evolve multi dimensional Turing Machines.

### Channel Assignment Problem (network)
A basic model used to find solutions to the channel assignment problem.

[//]: # (## 💾 Saving)
