Correlation Assisted Private Estimation (CAPE)
=====================================
Many applications of machine learning, such as human health research, involve processing private or sensitive information. Privacy concerns may impose significant hurdles to collaboration in scenarios where there are multiple sites holding data and the goal is to estimate properties jointly across all datasets. Conventional differentially private decentralized algorithms can provide strong privacy guarantees. However, the utility/accuracy of the joint estimates may be poor when the datasets at each site are small. We proposed a new framework, Correlation Assisted Private Estimation (CAPE), for designing privacy-preserving decentralized algorithms with much better accuracy guarantees in an honest-but-curious model. This repository includes MATLAB/Python codes for implementation of the CAPE protocol that can achieve the same utility as the pooled-data scenario in an honest-but-curious model. Please see the original paper: https://arxiv.org/abs/1904.10059 

The repository contains the codes for the following:
* visualizing the variation of effective $$\delta$$ resulting from the CAPE protocol with the number of colluding sites and the local noise variance. The plots empirically verify our claim that the CAPE protocol provides a better $$\delta$$ guarantee than the conventional decentralized differentially private scheme for a given noise level and $$\epsilon$$.
* simulating a neural network based classifier in the decentralized setting. The neural network is trained using a decentralized gradient descent that employs the CAPE protocol. We investigate the performance of the CAPE protocol and compare it with conventional scheme. We observe the variation in performance with varying number of samples and $$\epsilon$$ per iteration.

Feedback
------------

Please send bug reports, comments, or questions to [Hafiz Imtiaz](mailto:hafiz.imtiaz@outlook.com).
Contributions and extentions with new algorithms are welcome.
