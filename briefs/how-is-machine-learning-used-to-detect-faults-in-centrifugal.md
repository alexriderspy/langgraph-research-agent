# Machine Learning for Fault Detection in Centrifugal Pumps

## Methodology

A combination of t-SNE method and clustering techniques is then employed to detect novel faults [1]. Five key operational parameters were monitored: vibration, temperature, flow rate, pressure, and electrical current [2]. A dual-threshold labeling method was applied, combining fixed engineering limits with adaptive thresholds calculated as the 95th percentile of historical sensor values [2].

## Classification

Three machine learning classifiers - Random Forest, Extreme Gradient Boosting (XGBoost), and Support Vector Machine (SVM) - were trained to distinguish between normal operation, early warnings, and critical alerts [2].

## Validation

Finally, a test setup is used to validate this two-stage methodology on a centrifugal pump [1].

## Sources

1. [An Intelligent Approach to Detecting Novel Fault Classes for Centrifugal Pumps Based on Deep CNNs and Unsupervised Methods](https://arxiv.org/abs/2309.12765v1)
2. [AI-Powered Machine Learning Approaches for Fault Diagnosis in Industrial Pumps](https://arxiv.org/abs/2508.15550v1)
