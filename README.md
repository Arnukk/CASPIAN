# Deep Vision-Based Framework for Coastal Flood Prediction Under Climate Change Impacts and Shoreline Adaptations
[Areg Karapetyan](https://scholar.google.com/citations?user=MPNNFXMAAAAJ&hl=en&oi=ao), Aaron Chung Hin Chow, [Samer Madanat](https://scholar.google.com/citations?user=1OiQJ-EAAAAJ&hl=en&oi=ao) 

## Overview
In light of growing threats posed by climate change in general and sea level rise (SLR) in particular, the necessity for computationally efficient means to estimate and analyze potential coastal flood hazards has become increasingly pressing. Data-driven supervised learning methods serve as promising candidates that can dramatically expedite the process, thereby eliminating the ***computational bottleneck*** associated with traditional physics-based hydrodynamic simulators. Yet, the development of accurate and reliable coastal flood prediction models, especially those based on Deep Learning (DL) techniques, has been plagued with two major issues: (1) ***the scarcity of training data*** and (2) the high-dimensional output required for detailed inundation mapping.  To reinforce the arsenal of coastal inundation metamodeling techniques, we present a data-driven framework for synthesizing accurate and reliable DL-based coastal flood prediction models in ***low-resource learning settings***. The core idea behind the framework, which is graphically summarized in Fig. 1 below, is to recast the underlying multi-output regression problem as a computer vision task of translating a two-dimensional segmented grid into a matching grid with real-valued entries corresponding to water depths. 

<figure>
  <img src="https://i.postimg.cc/bwM7zHLg/Copy-of-methodology.jpg"
  alt="Schematic diagram of the proposed data-driven framework for training performant Deep Vision-based coastal flooding metamodels in low-data settings.">
  <figcaption>Figure 1: Schematic diagram of the proposed data-driven framework for training performant Deep Vision-based coastal flooding metamodels in low-data settings.</figcaption>
</figure>
</br></br>

The proposed methodology was tested on different neural networks, including two existing vision models: a fully transformer-based architecture ([SWIN-Unet](https://arxiv.org/abs/2105.05537)) and a Convolutional Neural Network (CNN) with additive attention gates ([Attention U-net](https://arxiv.org/abs/1804.03999)). Additionally, we introduce a deep CNN architecture, dubbed Cascaded Pooling and Aggregation Network (***CASPIAN***), stylized explicitly for the coastal flood prediction problem at hand. The introduced model, illustrated in Fig. 2 below, was designed with a particular focus on its compactness and practicality to cater to ***resource-constrained scenarios*** and ***accessibility aspects.*** Specifically, featuring as little as $0.36$ Mil. parameters and only a few main hyperparameters, CASPIAN can be easily trained and fine-tuned on a single GPU. On the current dataset, the performance of CASPIAN approached remarkably close to the results produced by the physics-based hydrodynamic simulator (on average, with 97\% of predicted floodwater levels having less than 10 cm. error), effectively reducing the computational cost of producing a flood inundation map from ***days to milliseconds***. 
<figure>
  <img src="https://i.postimg.cc/pr7ywLyd/architecture.jpg"
  alt="Schematic diagram of the proposed data-driven framework for training performant Deep Vision-based coastal flooding metamodels in low-data settings.">
  <figcaption>Figure 2: Detailed architecture of the proposed minimalistic CNN model, CASPIAN, for high-resolution coastal flood prediction under SLR and shoreline fortifications. The modulation blocks drawn as sketches in dotted outlines are optional and could be substituted by the output from the initial block. The operations followed by non-linear activation functions are marked with a blue border.</figcaption>
</figure>
</br></br>

Lastly, we provide a carefully curated [database of synthetic flood inundation maps](https://doi.org/10.7910/DVN/M9625R) of Abu Dhabi's coast for $174$ different shoreline protection scenarios. The maps were generated via a high-fidelity physics-based hydrodynamic simulator under a 0.5-meter SLR projection. The provided dataset, to the best of our knowledge, is the **first of its kind**, and thus can serve as a benchmark for evaluating future coastal flooding metamodels.

This repository contains the complete source code and data for reproducing the results reported in the paper. The proposed framework and models were implemented in `tensorflow.keras` (v 2.1). The weights of all the trained DL models are included.

The implementations of the SWIN-Unet and Attention U-net were adapted from the [keras-unet-collection](https://github.com/yingkaisha/keras-unet-collection) repository of [Yingkai (Kyle) Sha](https://github.com/yingkaisha).

For citing this work or the dataset, please use the below references.
```bibtex
@article{karapetyan2024,
  title={{Deep Vision-Based Framework for Coastal Flood Prediction Under Climate Change Impacts and Shoreline Adaptations}},
  author={Karapetyan,Areg and Chow, Chung Hin Aaron and Madanat, Samer},
  journal={arXiv preprint},
  year={2024}
}

@data{DVN/M9625R_2024,
author = {Karapetyan, Areg and Chow, Chung Hin Aaron and Madanat, Samer },
publisher = {Harvard Dataverse},
title = {{Simulated Flood Inundation Maps of Abu Dhabi's Coast Under Different Shoreline Protection Scenarios}},
year = {2024},
version = {V1},
doi = {10.7910/DVN/M9625R},
url = {https://doi.org/10.7910/DVN/M9625R}
}

```

## Repository Structure

TBA

