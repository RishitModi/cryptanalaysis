
# Energy-Based and Neurosymbolic methods for advanced Cryptanalysis

## Introduction

Cryptanalysis is the study of analysing cryptographic systems in order to recover secret information without prior knowledge of the key. While modern cryptographic algorithms are mathematically secure against classical attacks, practical implementations often leak information through physical side channels such as power consumption, timing behaviour, and electromagnetic radiation.

This project investigates cryptanalysis using a multi-layered approach that combines:

- Classical statistical cryptanalysis
- Side-channel leakage modelling
- Deep learning based profiling attacks
- Energy-based neurosymbolic key ranking

The objective is to demonstrate how machine learning techniques can exploit statistical dependencies between leakage traces and intermediate cryptographic computations to recover secret keys.

---

Modern block ciphers are designed to resist structural cryptanalysis. However, physical implementations of these algorithms exhibit data-dependent power consumption. This leakage can be modelled and analysed to recover secret keys.

The key challenges addressed in this project are:

- modelling realistic side-channel leakage
- learning leakage patterns using neural networks
- ranking key hypotheses using energy-based scoring

---

### Objectives

- Implement classical Vigenère cryptanalysis using statistical methods
- Generate synthetic side-channel traces using the Hamming Weight model
- Train CNN-based profiling attacks for key recovery
- Recover DES subkeys using intermediate value classification
- Develop an energy-based model for key candidate ranking

---

### Tools and Technologies

- Python
- PyTorch
- NumPy
- HDF5 dataset format
- Convolutional Neural Networks (CNNs)

# Energy-Based Cipher Classifier

## Purpose

The energy classifier is used to classify cipher-related patterns and distinguish between different cipher behaviours present in the project. Instead of directly predicting the secret key, the model learns to identify whether a given feature vector corresponds to a valid cipher transformation.

This module acts as the first stage of the pipeline by learning statistical representations of different cipher types and their structural behaviour.

---

## Cipher Types 

The project includes the following cipher categories:

- Advanced Encryption Standards(AES)
- Classical substitution-based cipher (Vigenère)
- Lightweight block cipher (SPECK32)
- Feistel-based block cipher (DES)

Each cipher exhibits distinct statistical and structural patterns. The energy classifier learns to differentiate these patterns.

---

## Theoretical Foundation

An Energy-Based Model defines a scalar function:

E(x, y)

where:

x = input feature vector (trace-derived or text-derived features)  
y = cipher class label

The model assigns:

Low energy → correct cipher class  
High energy → incorrect cipher class

---

## Classification Mechanism

During training, the model learns:

- statistical structure of ciphertext or trace features
- distributional differences between cipher types
- structural properties of transformations

This enables the classifier to distinguish between:

- classical polyalphabetic behaviour
- Feistel network leakage patterns
- ARX-based lightweight cipher leakage

---

## Training Objective

The loss function enforces:

E(x, y_true) < E(x, y_false)

This creates an energy margin between correct and incorrect cipher classes.

---

## Output Interpretation

The model produces an energy score for each cipher type.  
The predicted class is the one with minimum energy.

This allows:

- multi-class cipher classification
- statistical validation of cipher behaviour
- feature learning for downstream cryptanalysis

---

## Role in the Cryptanalysis Pipeline

The energy classifier:

- learns structural differences between cipher families
- provides feature-level understanding of cipher behaviour
- acts as a preprocessing and analysis stage before key recovery attacks

# Classical Cryptanalysis: Vigenère Cipher

## Background

The Vigenère cipher is a polyalphabetic substitution cipher defined as:

Ci = (Pi + Ki) mod 26

Its security depends on the secrecy of the key length and key characters.

---

## Key Length Detection

Two statistical techniques are used:

### Index of Coincidence (IC)

The IC measures the probability that two randomly selected letters are identical. English text has a higher IC than random text. By computing IC for different assumed key lengths, the correct key length can be estimated.

### Kasiski Examination

Repeated ciphertext patterns reveal periodic spacing. The greatest common divisor of these spacings provides candidate key lengths.

---

## Frequency Analysis

After determining the key length, the ciphertext is divided into columns. Each column behaves like a Caesar cipher. By comparing letter frequency distributions with English frequencies, the shift for each column is determined and the key is reconstructed.

---

## Outcome

- Successful recovery of the key
- Demonstration of classical statistical cryptanalysis
- Baseline comparison for modern machine learning based attacks

# Side-Channel Analysis of DES and SPECK32

## What is Side-Channel Analysis

Side-Channel Analysis (SCA) exploits physical leakage from cryptographic implementations rather than mathematical weaknesses. Power consumption is data-dependent and can reveal information about intermediate computations.

---

## Leakage Model

The Hamming Weight (HW) model is used:

HW(x) = number of bits set to 1 in x

Power consumption is assumed to be proportional to the Hamming Weight of intermediate values.

---

## SPECK32: Synthetic Trace Generation

Synthetic traces are generated by recording HW values of intermediate operations during encryption:

- rotation output
- modular addition result
- XOR with round key
- final mixing stage

Each round produces four leakage samples.

Trace length = 22 × 4 = 88 samples

Dataset characteristics:

- 10,000 traces
- fixed secret key
- random plaintexts
- Gaussian noise added
- random desynchronization applied

This simulates realistic measurement conditions.

---

## CNN-Based Key Recovery for SPECK32

A Convolutional Neural Network is trained to learn the mapping:

trace → key class

Key rank is used as the evaluation metric.  
Rank 0 indicates successful key recovery.

---

## DES S-box Side-Channel Attack

Instead of predicting the key directly, the model predicts the S-box output (16 classes).

Key recovery procedure:

1. Compute S-box output for each key hypothesis
2. Accumulate log-likelihoods across traces
3. Select the key with maximum likelihood

This follows the template attack methodology with deep learning feature extraction.

---

## Advantages of Deep Learning in SCA

- automatic detection of leakage points
- robustness to noise and misalignment
- no manual feature engineering required
