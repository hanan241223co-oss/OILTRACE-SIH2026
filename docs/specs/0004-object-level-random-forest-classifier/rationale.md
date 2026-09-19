# Rationale: 0004 Object Level Random Forest Slick Classifier

## Context

Marine oil spill surveillance with synthetic aperture radar operates under challenging environmental conditions. Mineral oil discharges dampen short capillary ocean waves, creating dark areas with reduced backscatter. However, numerous natural ocean phenomena also suppress capillary waves, producing identical dark patches in radar imagery. These natural lookalikes include calm ocean water in low wind pools, biogenic slicks produced by plankton or algae blooms, rain cells, internal ocean waves, and radar shadow corridors cast behind metallic ship hulls.

Pixel level classifiers analyze each radar measurement independently or across small local pixel neighborhoods. Because the decibel backscatter values of a biogenic slick or ship shadow can match those of an illegal crude oil discharge, pixel classifiers suffer from high false alarm rates. When deployed in operational surveillance, false alarm rates overwhelm maritime response teams and erode confidence in automated alert systems.

To resolve this challenge, the OILTRACE detection pipeline adopts an object based architecture. In Feature 9, candidate low backscatter ocean regions are segmented into discrete geographic polygons. In Feature 10, each polygon is enriched with 17 physical and geometric metrics describing shape morphology, radar backscatter distributions, interfacial edge sharpness, and vessel proximity.

The decision now facing the system is selecting the object level classification algorithm, training methodology, and probability calibration strategy. The chosen model must consume these 17 dimensional feature vectors, generalize across varying sea states, handle severe class imbalance between rare oil spills and abundant natural lookalikes, and output reliable, calibrated confidence scores.

## Options considered

### Option 1: Scikit-learn Random Forest with Balanced Class Weighting and Canonical 17 Features (Chosen)

Train an ensemble of 150 randomized decision trees using scikit-learn `RandomForestClassifier` directly on the 17 canonical features from Feature 10. Compensate for class imbalance using `class_weight='balanced'`, which inversely weights sample losses based on class prevalence. Calibrate the operating decision threshold at 0.60 to prioritize operational precision.

**Pros**:
* Exceptional resistance to overfitting on moderate tabular datasets through bootstrap aggregation and random feature subset selection.
* Natively handles non linear interactions between disparate feature types such as dimensionless shape metrics, decibel radar statistics, and geographic distances in meters without requiring feature scaling or normalization.
* Fast training and sub millisecond inference per polygon, suitable for rapid operational scene processing.
* Built in Gini feature importance ranking provides physical interpretability for marine operators.
* Compact serialization footprint under 2 megabytes using standard joblib tooling.

**Cons**:
* Requires choosing an operating decision threshold to balance precision and recall rather than relying on default 0.50 cutoff.

### Option 2: Gradient Boosted Decision Trees (LightGBM or XGBoost)

Train an iterative gradient boosted decision tree ensemble that minimizes classification loss by sequentially correcting residual errors.

**Pros**:
* High predictive capacity that can occasionally extract fractional gains in area under the curve.
* Native handling of missing values.

**Cons**:
* Adds external library dependencies (XGBoost or LightGBM) outside the existing core project environment.
* Significantly more sensitive to hyperparameter tuning and prone to overfitting on small datasets with limited positive oil spill samples.
* Tree construction is sequential rather than parallel, offering fewer operational advantages given our small candidate count per scene.

### Option 3: Support Vector Classifier with Radial Basis Function Kernel

Train a margin maximizing hyperplane separator using scikit-learn `SVC` with radial basis functions.

**Pros**:
* Strong theoretical foundations for binary boundary separation in high dimensional spaces.

**Cons**:
* Requires strict feature standard normalization because input features have wildly varying scales and physical units (ratios, decibels, meters, pixel areas).
* Slow quadratic training time scaling with sample count.
* Does not provide direct calibrated class probabilities without expensive Platt scaling cross validation.
* Lacks native feature importance inspection for operational dossiers.

## Decision

**Chosen option**: Option 1: Scikit-learn Random Forest with Balanced Class Weighting and Canonical 17 Features.

We adopt the Random Forest classifier trained on the 17 canonical features with balanced class weighting, 150 trees, max depth of 12, and a high precision decision threshold of 0.60.

## Rationale

Marine oil spill surveillance demands a classifier that is robust, fast, and physically interpretable. Random Forest aligns perfectly with the project stack (scikit-learn 1.3.2) and the characteristics of our tabular candidate feature matrix. 

Because candidate features combine geometric morphology, radar backscatter distributions, boundary gradients, and spatial proximity, tree based models handle these unnormalized mixed scale distributions seamlessly. The balanced class weighting strategy ensures the model learns discriminative boundaries for rare mineral oil slicks without getting swamped by hundreds of lookalike candidates. 

Setting an operational decision threshold of 0.60 suppresses borderline false alarms, fulfilling the primary requirement of marine surveillance authorities: when an alert is generated, it must carry high confidence.

## Follow-up

* [ ] Re evaluate hyperparameter grid when additional annotated satellite SAR scenes become available in future dataset expansions.
