<!-- # Project Overview

* We built an Energy Classifier model that can automatically detect which encryption algorithm — AES, DES, Vigenère, or Speck32 — was used just by analyzing the ciphertext.

* The classifier doesn’t rely on knowing the encryption key; instead, it studies statistical and structural patterns present in the ciphertext to make predictions.

* Our approach focuses on using deep learning to extract hidden representations from encrypted data, helping the model learn how different algorithms leave distinct “signatures.”

* Alongside classification, we explored Side-Channel Analysis (SCA), a practical attack method that uses information like power consumption or electromagnetic leaks to recover secret keys.

* We built a basic SCA model for DES, where the model learns to identify key-related leakage from traces, and tested its ability to recover or rank the correct key.

* Also, generated synthetic data for Speck32 using Hamming weight.

* The project includes the process of generating datasets, training both the classifier and SCA model, and evaluating results using accuracy and key rank metrics.

###

# Why This Matters

* This project helps us understand the real-world security behaviour of classical and modern ciphers when analysed using deep learning and side-channel techniques.

* By combining cipher classification and key recovery attempts, we get insights into how secure each encryption method is.

* It shows that older ciphers like DES and Vigenère are easier to analyze or attack, while modern ones like AES remain highly resistant — even to advanced models.

* We learn how SCA reveals vulnerabilities not through mathematical weakness but through physical leakage, showing a different layer of cryptanalysis.

* The project bridges classical cryptography, modern deep learning, and hardware-level analysis, giving a complete picture of encryption security from multiple perspectives.

* Overall, it connects theory, experimentation, and attack simulation, helping us understand how encryption systems can be analysed, compared, and made more secure in practice. -->


## Introduction

Modern cryptography is designed to produce ciphertext that appears statistically random, making it extremely difficult to identify the underlying encryption algorithm without the key. However, different ciphers often leave subtle structural and statistical traces that can be learned by deep learning models.

This project explores whether machine learning — specifically energy-based models, CNNs, and transformers — can:

-  Identify which encryption algorithm generated a ciphertext

-  Learn leakage patterns from side-channel traces

-  Recover or rank secret keys without brute force

We study both algorithm-level security and implementation-level leakage, giving a complete, practical view of cryptanalysis.


## Objectives

Build a cipher classifier that predicts
- AES  
- DES  
- Vigenère 
- Speck32
using ciphertext only

Implement a Side-Channel Analysis (SCA) model for DES

Generate synthetic leakage data using Hamming Weight for Speck32

Evaluate:

Classification accuracy

Key rank performance

Compare classical vs modern cipher robustness against deep learning