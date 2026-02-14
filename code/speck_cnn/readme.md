# Side Channel Attack on Speck32

* This part of project generates synthetic traces to study how encryption leaks information.

* Each trace records **Hamming Weight** of intermediate operations during encryption.

* random **Gaussian Noise** is added and made datset desynchronized to make it close to real traces.

* The model used is 1D Convolutional Neural Networks (CNNs)

* The evaluation is done using key rank, not accuracy.

* In practice we only have 88 time-samples per trace (22 rounds x 4 operations), so diversity is limited. Real Hardware traces are needed for realistic outputs

* Thus, we are getting very high accuracy.