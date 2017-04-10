
# 4 Models

<!-- toc orderedList:0 depthFrom:1 depthTo:6 -->

* [4 Models](#4-models)
  * [4.1 Scaling](#41-scaling)
    * [4.1.1 Dimensionless Variables](#411-dimensionless-variables)
    * [4.1.2 Dimensionless Numbers](#412-dimensionless-numbers)
    * [4.1.3 A Scaling for Vanishing Initial Condition](#413-a-scaling-for-vanishing-initial-condition)
  * [4.2 Evolution of a Population](#42-evolution-of-a-population)
    * [4.2.1 Exponential Growth](#421-exponential-growth)
    * [4.2.2 Logistic Growth](#422-logistic-growth)
  * [4.3 Compound Interest and Inflation](#43-compound-interest-and-inflation)
  * [4.4 Newton’s Law of Cooling](#44-newtons-law-of-cooling)
  * [4.5 Radioactive Decay](#45-radioactive-decay)
    * [4.5.1 Deterministic Model](#451-deterministic-model)
    * [4.5.2 Stochastic Model](#452-stochastic-model)
    * [4.5.3 Relation Between Stochastic and Deterministic Models](#453-relation-between-stochastic-and-deterministic-models)
    * [4.5.4 Generalization of the Radioactive Decay Modeling](#454-generalization-of-the-radioactive-decay-modeling)
  * [4.6 Chemical Kinetics](#46-chemical-kinetics)
    * [4.6.1 Irreversible Reaction of Two Substances](#461-irreversible-reaction-of-two-substances)
    * [4.6.2 Reversible Reaction of Two Substances](#462-reversible-reaction-of-two-substances)
    * [4.6.3 Irreversible Reaction of Two Substances into a Third](#463-irreversible-reaction-of-two-substances-into-a-third)
    * [4.6.4 A Biochemical Reaction](#464-a-biochemical-reaction)
  * [4.7 Spreading of Diseases](#47-spreading-of-diseases)
  * [4.8 Predator-Prey Models in Ecology](#48-predator-prey-models-in-ecology)
  * [4.9 Decay of Atmospheric Pressure with Altitude](#49-decay-of-atmospheric-pressure-with-altitude)
    * [4.9.1 The General Model](#491-the-general-model)
    * [4.9.2 Multiple Atmospheric Layers](#492-multiple-atmospheric-layers)
    * [4.9.3 Simplifications](#493-simplifications)
  * [4.10 Compaction of Sediments](#410-compaction-of-sediments)
  * [4.11 Vertical Motion of a Body in a Viscous Fluid](#411-vertical-motion-of-a-body-in-a-viscous-fluid)
    * [4.11.1 Overview of Forces](#4111-overview-of-forces)
    * [4.11.2 Equation of Motion](#4112-equation-of-motion)
    * [4.11.3 Terminal Velocity](#4113-terminal-velocity)
    * [4.11.4 A Crank–Nicolson Scheme](#4114-a-cranknicolson-scheme)
    * [4.11.5 Physical Data](#4115-physical-data)
    * [4.11.6 Verification](#4116-verification)
    * [4.11.7 Scaling](#4117-scaling)
  * [4.12 Viscoelastic Materials](#412-viscoelastic-materials)
  * [4.13 Decay ODEs from Solving a PDE by Fourier Expansions](#413-decay-odes-from-solving-a-pde-by-fourier-expansions)
  * [4.14 Exercises](#414-exercises)
    * [Exercise 4.1: Radioactive decay of Carbon-14](#exercise-41-radioactive-decay-of-carbon-14)
    * [Exercise 4.2: Derive schemes for Newton’s law of cooling](#exercise-42-derive-schemes-for-newtons-law-of-cooling)
    * [Exercise 4.3: Implement schemes for Newton’s law of cooling](#exercise-43-implement-schemes-for-newtons-law-of-cooling)
    * [Exercise 4.4: Find time of murder from body temperature](#exercise-44-find-time-of-murder-from-body-temperature)
    * [Exercise 4.5: Simulate an oscillating cooling process](#exercise-45-simulate-an-oscillating-cooling-process)
    * [Exercise 4.6: Simulate stochastic radioactive decay](#exercise-46-simulate-stochastic-radioactive-decay)
    * [Exercise 4.8: Simulate a simple chemical reaction](#exercise-48-simulate-a-simple-chemical-reaction)
    * [Exercise 4.9: Simulate an n-th order chemical reaction](#exercise-49-simulate-an-n-th-order-chemical-reaction)
    * [Exercise 4.10: Simulate a biochemical process](#exercise-410-simulate-a-biochemical-process)
    * [Exercise 4.11: Simulate spreading of a disease](#exercise-411-simulate-spreading-of-a-disease)
    * [Exercise 4.12: Simulate predator-prey interaction](#exercise-412-simulate-predator-prey-interaction)
    * [Exercise 4.13: Simulate the pressure drop in the atmosphere](#exercise-413-simulate-the-pressure-drop-in-the-atmosphere)
    * [Exercise 4.14: Make a program for vertical motion in a fluid](#exercise-414-make-a-program-for-vertical-motion-in-a-fluid)
    * [Exercise 4.15: Simulate parachuting](#exercise-415-simulate-parachuting)
    * [Exercise 4.16: Formulate vertical motion in the atmosphere](#exercise-416-formulate-vertical-motion-in-the-atmosphere)
    * [Exercise 4.17: Simulate vertical motion in the atmosphere](#exercise-417-simulate-vertical-motion-in-the-atmosphere)
    * [Exercise 4.18: Compute y = |x| by solving an ODE](#exercise-418-compute-y-x-by-solving-an-ode)
    * [Exercise 4.19: Simulate fortune growth with random interest rate](#exercise-419-simulate-fortune-growth-with-random-interest-rate)
    * [Exercise 4.20: Simulate a population in a changing environment](#exercise-420-simulate-a-population-in-a-changing-environment)
    * [Exercise 4.21: Simulate logistic growth](#exercise-421-simulate-logistic-growth)
    * [Exercise 4.22: Rederive the equation for continuous compound interest](#exercise-422-rederive-the-equation-for-continuous-compound-interest)
    * [Exercise 4.23: Simulate the deformation of a viscoelastic material](#exercise-423-simulate-the-deformation-of-a-viscoelastic-material)

<!-- tocstop -->


## 4.1 Scaling

### 4.1.1 Dimensionless Variables

### 4.1.2 Dimensionless Numbers

### 4.1.3 A Scaling for Vanishing Initial Condition

## 4.2 Evolution of a Population

### 4.2.1 Exponential Growth

### 4.2.2 Logistic Growth

## 4.3 Compound Interest and Inflation

## 4.4 Newton’s Law of Cooling

## 4.5 Radioactive Decay

* http://en.wikipedia.org/wiki/Radioactive_decay

### 4.5.1 Deterministic Model

### 4.5.2 Stochastic Model

* http://en.wikipedia.org/wiki/Bernoulli_trial

### 4.5.3 Relation Between Stochastic and Deterministic Models

* http://en.wikipedia.org/wiki/Exponential_distribution

### 4.5.4 Generalization of the Radioactive Decay Modeling

## 4.6 Chemical Kinetics

### 4.6.1 Irreversible Reaction of Two Substances

### 4.6.2 Reversible Reaction of Two Substances

### 4.6.3 Irreversible Reaction of Two Substances into a Third

* https://en.wikipedia.org/wiki/Law_of_mass_action

### 4.6.4 A Biochemical Reaction

## 4.7 Spreading of Diseases

* https://en.wikipedia.org/wiki/Michaelis-Menten_kinetics

## 4.8 Predator-Prey Models in Ecology

## 4.9 Decay of Atmospheric Pressure with Altitude

### 4.9.1 The General Model

### 4.9.2 Multiple Atmospheric Layers

### 4.9.3 Simplifications

* http://en.wikipedia.org/wiki/Density_of_air

## 4.10 Compaction of Sediments

## 4.11 Vertical Motion of a Body in a Viscous Fluid

### 4.11.1 Overview of Forces

* <a href="http://en.wikipedia.org/wiki/Drag_(physics)">http://en.wikipedia.org/wiki/Drag_(physics)</a>

### 4.11.2 Equation of Motion

### 4.11.3 Terminal Velocity

### 4.11.4 A Crank–Nicolson Scheme

### 4.11.5 Physical Data

### 4.11.6 Verification

* http://en.wikipedia.org/wiki/Drag_coefficient

### 4.11.7 Scaling

## 4.12 Viscoelastic Materials

* https://en.wikipedia.org/wiki/Kelvin-Voigt_material

## 4.13 Decay ODEs from Solving a PDE by Fourier Expansions

## 4.14 Exercises

### Exercise 4.1: Radioactive decay of Carbon-14

* http://en.wikipedia.org/wiki/Carbon-14

### Exercise 4.2: Derive schemes for Newton’s law of cooling

### Exercise 4.3: Implement schemes for Newton’s law of cooling


```python
def test_discrete_solution():
    """
    Compare the numerical solution with an exact solution
    of the scheme when the T_s is constant.
    """
    T_s = 10
    T0 = 2
    k = 1.2
    dt = 0.1 # can use any mesh
    N_t = 6 # any no of steps will do
    t_end = dt*N_t
    t = np. linspace(0, t_end, N_t+1)
    for theta in [0, 0.5, 1, 0.2]:
        u, t = cooling(T0, k, lambda t: T_s , t_end, dt, theta)
        A = (1 - (1-theta)*k*dt)/(1 + theta*k*dt)
        u_discrete_exact = T_s + (T0-T_s)*A**(np. arange(len(t)))
        diff = np. abs(u - u_discrete_exact). max()
        print 'diff computed and exact discrete solution:' , diff
        tol = 1E-14
        success = diff < tol
        assert success, 'diff=%g' % diff
```


```python

```

### Exercise 4.4: Find time of murder from body temperature

### Exercise 4.5: Simulate an oscillating cooling process

### Exercise 4.6: Simulate stochastic radioactive decay


```python
# Given lambda_ , dt , N
import numpy as np
uniform = np. random. uniform(N)
Bernoulli_trials = np. asarray(uniform < lambda_*dt, dtype=np. int)
d = Bernoulli_trials. size
```

### Exercise 4.8: Simulate a simple chemical reaction

### Exercise 4.9: Simulate an n-th order chemical reaction

### Exercise 4.10: Simulate a biochemical process

### Exercise 4.11: Simulate spreading of a disease

### Exercise 4.12: Simulate predator-prey interaction

### Exercise 4.13: Simulate the pressure drop in the atmosphere

### Exercise 4.14: Make a program for vertical motion in a fluid

### Exercise 4.15: Simulate parachuting

### Exercise 4.16: Formulate vertical motion in the atmosphere

* http://en.wikipedia.org/wiki/Parachuting

### Exercise 4.17: Simulate vertical motion in the atmosphere

* http://en.wikipedia.org/wiki/Felix_Baumgartner

### Exercise 4.18: Compute y = |x| by solving an ODE

### Exercise 4.19: Simulate fortune growth with random interest rate


```python
import random
def new_interest_rate(p_n, dp=0.5):
    r = random. random() # uniformly distr. random number in [0,1)
    if 0 <= r < 0.25:
        p_np1 = p_n + dp
    elif 0.25 <= r < 0.5:
        p_np1 = p_n - dp
    else:
        p_np1 = p_n
    return (p_np1 if 1 <= p_np1 <= 15 else p_n)
```

### Exercise 4.20: Simulate a population in a changing environment

### Exercise 4.21: Simulate logistic growth

### Exercise 4.22: Rederive the equation for continuous compound interest

### Exercise 4.23: Simulate the deformation of a viscoelastic material


```python

```
