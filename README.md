# Energy-Based and Neurosymbolic Methods for Advanced Cryptanalysis

[Project Repo](https://github.com/AnuushkaY/EnergyBased-and-Neurosymbolic-Methods-for-Advanced-CrypytAnalysis)  
[Documentation](https://anuushkay.github.io/Cryptanalysis_Documentation/)

---

## Overview

This project explores cryptanalysis using a multi-layered approach that combines:

- Classical statistical attacks  
- Side-channel leakage modelling  
- Deep learning profiling attacks  
- Energy-based neurosymbolic cipher classification  

Instead of attacking keys directly, the pipeline first **classifies the cipher type** and then applies the appropriate **key-recovery strategy**.

---

## Objectives

| Component | Goal |
|-----------|------|
Classical Cryptanalysis | Break Vigenère using IC, Kasiski, frequency analysis |
Side-Channel Modelling | Generate synthetic traces using Hamming Weight |
Deep Learning SCA | CNN-based key recovery for DES and SPECK32 |
Energy Model | Classify cipher families using energy scoring |

---

# Energy-Based Cipher Classifier

## Purpose

The energy classifier identifies the **cipher family** from feature vectors before key recovery.  
It learns structural patterns of different ciphers rather than predicting keys.

---

## Cipher Classes

| Cipher | Structure | Feature Type |
|--------|----------|--------------|
AES | SPN | Statistical / structural |
Vigenère | Polyalphabetic | Frequency patterns |
DES | Feistel | S-box leakage |
SPECK32 | ARX | HW trace patterns |

---

## Role in Pipeline

- Learns structural differences between cipher families  
- Acts as preprocessing stage  
- Guides selection of correct cryptanalytic attack  

---

# Classical Cryptanalysis: Vigenère

## Key Length Detection

| Method | Principle |
|--------|-----------|
Index of Coincidence | Detects similarity to English distribution |
Kasiski Examination | Uses repeated pattern spacing |

---

## Key Recovery

- Split ciphertext into columns  
- Treat each column as Caesar cipher  
- Apply frequency analysis  

Provides a **baseline classical attack**

---

# Side-Channel Analysis (DES & SPECK32)

## Leakage Model

Hamming Weight:

HW(x) = number of 1 bits in x

Power ∝ HW(intermediate value)

---

## SPECK32 Synthetic Dataset

CNN learns:

trace -> key class

Metric: **Key Rank (Rank 0 = success)**

---

## DES S-box Profiling Attack

Model predicts **S-box output (16 classes)**

Key recovery:

1. Compute S-box output for each key guess  
2. Accumulate log-likelihoods  
3. Select maximum likelihood key  

Deep learning acts as **feature extractor + template attack**

---

# Future Scope

| Challenge | Direction |
|-----------|-----------|
Synthetic traces | Use real hardware power traces |
Noise robustness | Apply masking-aware training |
Model scaling | Transformers for long traces |
Higher-order attacks | Capture multivariate leakage |
Generalisation | Cross-device evaluation |

---

## Tech Stack

- Python  
- PyTorch  
- NumPy  
- HDF5  
- CNN Architectures  

## Mentors  

- [@Afreen Kazi](https://github.com/Afreen-Kazi-1)  
- [@Ghruank Kothare](https://github.com/Ghruank)
