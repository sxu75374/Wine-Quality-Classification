<div id="top"></div>

# Wine Quality Classification


<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#about-the-project">About The Project</a></li>
    <li><a href="#contents">Contents</a></li>
    <li><a href="#screenshots">Screenshots</a></li>
    <li><a href="#built-with">Built With</a></li>
      <ul>
          <li><a href="#installation">Installation</a></li>
      </ul>
    <li><a href="#author">Author</a></li>
    <li><a href="#license">License</a></li>
  </ol>
</details>

## About The Project

This project I choose the [Wine Quality Dataset (White)](https://archive.ics.uci.edu/ml/datasets/Wine+Quality), which the original dataset is from the UCI Machine Learning Repository. The data is from the real world and each of the features will decide the quality of the wine quality. But, in my experiment, part of the dataset has been pre-processed. The dataset includes **11 features** (fixed acidity, volatile acidity, var1*, var2*, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates and var3*, where var1*, var2*, var3* are the three features that has been pre-processed from the alcohol, residual sugar and citric acid) and **3 classes** (class 1 is good, class 2 is medium, and class 3 is bad quality).

The goal is that I use some pre-processing techniques on the wine training and testing set to process the data, then implement several classification algorithms by validation or cross-validation to find good classification models, and finally use the best models I get to classify the test set to see the model performance for the unknown dataset by the accuracy, macro f1 score and confusion matrix three measurements. Because the dataset is biased, I compare the oversampling and undersampling meethods to see the different performance.

## Contents
In the main file `Main.py`, I’m going to try different data pre-processing methods to process the data, includes Standardization/Normalization, dataset balance, feature engineering.
**Preprocessing**: based on the pre-processed dataset, I will use six different machine learning algorithms to do the classification for the wine quality dataset which modified from the dataset from UCI machine learning Repository. Then, I create a probability-based trivial system and a baseline system by default perceptron with standardized data. For the six classification algorithms I chose are Linear Discriminant Analysis, K-Nearest Neighbors, Support Vector Machine, Stochastic Gradient Descend Classifier, Random Forest and Multi-Layer Perceptron. 
**Model selection**: I do the model selection by the cross-validation to get the best model and do the farther test by the test set.
**Performance Comparison**: The main performance measures for the classification results comparison are: Accuracy, Macro f1 score and Confusion Matrix. Base on the classification result, I do the comparison with the trivial system and baseline system and do the comparison between each of the algorithms to do the farther analysis. 

## Screenshots
<br />
<div align="center">
  <img src="screenshots/screenshot1.png" alt="screenshot1" width="550" height="400">
  <img src="screenshots/screenshot2.png" alt="screenshot2" width="550" height="400">
  <img src="screenshots/screenshot3.png" alt="screenshot3" width="530" height="400">
</div>


## Built With
- [Python 3.7.4](https://www.python.org/downloads/release/python-374/)


### Installation
This code built and tested with Python 3.7.4, included package scikit-learn 0.24.2, pandas 1.2.4, numpy 1.20.2, scipy 1.6.3, matplotlib 3.4.1, and imblearn 0.8.0.

<!--## further improvement-->

## Author

**Shuai Xu** | University of Southern California

[Profile](https://github.com/sxu75374) - <a href="mailto:sxu75374@usc.edu?subject=Nice to meet you!&body=Hi Shuai!">Email</a>

Project Link: [https://github.com/sxu75374/Wine-Quality-Classification](https://github.com/sxu75374/Wine-Quality-Classification)

<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.md` for more information.

<p align="right">[<a href="#top">back to top</a>]</p>
