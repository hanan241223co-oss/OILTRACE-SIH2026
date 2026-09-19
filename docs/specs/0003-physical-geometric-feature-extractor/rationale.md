# Rationale: 0003 Physical & Geometric Object Feature Extractor

## Context

In satellite synthetic aperture radar surveillance, mineral oil slicks dampen capillary ocean waves, appearing as dark patches of reduced radar backscatter. However, natural ocean phenomena such as biogenic algal films, calm water pools in low wind areas, rain cells, and grease ice also dampen capillary waves, generating lookalikes that produce identical low decibel pixel values.

Pixel level classifiers evaluate each radar measurement in isolation, making them incapable of distinguishing between mineral oil and natural lookalikes. To overcome this limitation, the detection architecture uses object based image analysis. In Stage 1, candidate dark spots are segmented into discrete geographic polygons. In this intermediate stage, the pipeline extracts physical and geometric properties from each candidate polygon.

Mineral oil discharges typically form elongated, narrow trails following ship paths or ocean currents, with sharp interfacial phase boundaries and strong physical damping ratios between 3 and 10 decibels. Natural biogenic films, by contrast, display irregular diffuse boundaries, lower wave damping contrast, and broad organic spread. By extracting a standardized 17 dimensional feature vector capturing shape geometry, radar backscatter distribution, boundary edge sharpness, and vessel proximity, the system equips downstream machine learning models to separate illegal oil spills from natural sea surface variations.

## Options considered

### Option 1: Multi dimensional physical and geometric feature suite with precomputed scene gradients (Chosen)

Extract 17 complementary features across 4 distinct physical categories: geometric shape (area, perimeter, elongation, circularity, complexity ratio, pixel fractal dimension, extent), radar backscatter (mean VV, mean VH, peak min VV, polarimetric difference, texture standard deviation, local ocean damping), boundary gradient sharpness (mean and peak 95th percentile Sobel gradient sampled from precomputed scene gradient along valid boundary rings), and vessel context (distance to vessel, touches suspect vessel).

**Pros**:
* Captures the full multi physical profile of oil slicks compared to biological lookalikes.
* Precomputing the scene wide Sobel gradient once runs boundary analysis in $O(1)$ per candidate rather than full scene re-convolution, processing thousands of polygons in under 0.5 seconds.
* Pixel unit fractal calculation with $A_{px} \ge 30$ completely avoids division by zero or negative logarithms.
* Dual container design provides intuitive Python object access while enabling fast tabular matrix export for scikit-learn.

**Cons**:
* Requires allocating one additional float32 gradient raster buffer during scene feature extraction.

### Option 2: Basic geometric shape descriptors only

Compute standard bounding box dimensions, pixel counts, and perimeter without calculating boundary gradient sharpness or local ocean background contrast.

**Pros**:
* Fast execution using basic polygon properties.
* Simple implementation with fewer moving parts.

**Cons**:
* Fails to separate natural low wind pools from calm oil slicks because both can exhibit similar macro shapes.
* Discards valuable polarimetric VV/VH contrast and physical damping ratios.

### Option 3: Deep convolutional patch feature embeddings

Extract raw image patches centered around each candidate polygon and feed them through a pretrained convolutional neural network to produce deep feature embeddings.

**Pros**:
* Learns hierarchical visual representations directly from data.

**Cons**:
* Heavy computational overhead and dependency on GPU acceleration.
* Lacks physical interpretability: surveillance operators cannot inspect why a feature vector triggered an alarm.
* Prone to overfitting on small SAR training datasets.

## Decision

**Chosen option**: Option 1: Multi dimensional physical and geometric feature suite with precomputed scene gradients.

We extract a calibrated 17 dimensional feature vector combining geometric morphology, radar backscatter statistics, interfacial boundary sharpness, and vessel proximity for every candidate dark spot polygon.

## Rationale

Marine radar surveillance requires transparent, physically grounded features. Mineral oil slicks possess distinct physical properties: high damping contrast against local background water, high elongation when discharged by moving ships or advected by currents, low internal texture standard deviation due to uniform capillary wave suppression, and sharp interfacial boundaries resulting from oil water surface tension. Natural biogenic slicks, conversely, exhibit diffuse edges and lower damping ratios. Option 1 models these exact physical phenomena, providing interpretable and robust inputs for random forest classification.

## Follow-up

* [ ] Train Feature 11 (Object Level Random Forest Slick Classifier) using these extracted 17 dimensional feature vectors.
* [ ] Evaluate feature importance scores (MDI and permutation importance) to verify which features contribute most to lookalike suppression.
