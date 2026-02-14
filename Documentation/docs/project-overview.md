# Welcome to our project on Advanced Cryptanalysis!

***

## Project Idea

Modern cryptography forms the backbone of digital security, but the analysis and breaking of complex ciphers remain a significant challenge. This project explores a novel approach to cryptanalysis by combining two powerful AI paradigms: **Energy-Based Models (EBMs)** and **Side Channel Attacks**.

The goal is to leverage the pattern-recognition strengths of neural networks to identify statistical weaknesses in ciphertexts, while using the logical reasoning of symbolic AI to enforce the strict mathematical rules of the cryptographic algorithm.The main idea is to test security systems across the globe and whether they are correctly encrypting data using AES or not and trying  that if its not AES then key recovery is possible using various techniques.

## Overview

In this project:

* We will start by targeting classification of the most common ciphers : AES,DES,Vignere, and Speck-32.
* An Energy-Based Model will be trained to act as a classifier,to classify the type of encryption based purely on ciphertext.
* Once identified,we will use different methods to try and aim for key recovery,again using only the ciphertext.
*    *AES -    optimum condition of security.
    *DES - key recovery using a SCA model.
    *Speck - key recovery using another SCA model.
    *Vignere - key recovery using Energy Transformers encoder-decoder.
    
**Team Members** - Anushka Yadav, Rishit Modi
**Mentors** - Ghruank Kothare, Afreen