# Rationale: 0002 Adaptive Dark Spot Candidate Segmentation

## Context

Satellite synthetic aperture radar (SAR) instruments detect marine oil slicks because surface oil films dampen capillary ocean waves, creating dark patches of reduced radar backscatter. However, classical pixel level classification using fixed global decibel thresholds suffers from severe false alarms. Ocean wind speeds vary across radar swaths, and incidence angle decay causes background sea clutter to vary by up to ten decibels from near range to far range. A fixed threshold flags large areas of natural low wind water in one part of an image while completely missing true oil slicks in another.

In addition, single pixel noise (speckle) creates thousands of isolated dark pixels across the sea surface. Classifying individual pixels in isolation produces fragmented, noisy masks that cannot capture spatial slick geometry or physical boundary sharpness. To eliminate false alarms, remote sensing workflows require a two stage object based image analysis architecture. Stage 1 must segment coherent candidate dark spot regions with high recall, ensuring true spills are retained while packaging them as discrete geometric objects for Stage 2 machine learning classification.

## Options considered

### Option 1: Adaptive moving window local mean with relative damping contrast (Chosen)

Compute local ambient sea clutter continuously across the radar scene using a moving window box filter (101 by 101 pixels), normalized over valid ocean pixels with a global median fallback for low sample coastal zones. Flag candidate pixels where radar backscatter drops at least 2.5 decibels below local ocean clutter and below a negative 20.0 decibel ceiling.

**Pros**:
* Adapts dynamically to wind variations and incidence angle decay across the radar swath.
* Separates local wave damping from regional background shifts.
* Extremely fast execution using box filter convolution.

**Cons**:
* Requires adequate valid ocean pixels around image borders to compute stable moving averages.

### Option 2: Constant global decibel thresholding

Apply fixed cutoff values to raw VV and VH backscatter channels across the entire scene (for example, marking any pixel where VV is below negative 23.0 decibels as a dark spot).

**Pros**:
* Trivial to implement with minimal computational overhead.
* Requires no spatial convolution or neighborhood padding.

**Cons**:
* Fails when wind speed varies across the scene, creating massive false alarms in calm bays.
* Misses true oil slicks located in rougher seas where ambient clutter is higher.

### Option 3: Multi scale contrast pyramid

Compute moving averages and damping ratios across multiple pyramid scales (for example, 25 by 25, 51 by 51, and 101 by 101 pixel windows) and combine them with multi scale logical operations.

**Pros**:
* Captures both fine narrow oil trails and broad diffuse candidate patches.

**Cons**:
* Triples the convolution overhead and memory footprint.
* Introduces complex parameter tuning without clear performance gains over a well tuned 101 by 101 window.

## Decision

**Chosen option**: Option 1: Adaptive moving window local mean with relative damping contrast.

The pipeline will calculate local ambient ocean clutter using an efficient box filter normalized over valid ocean pixels, extracting candidate dark spot regions with high recall based on local damping contrast and absolute backscatter limits.

## Rationale

Marine oil slicks do not have a constant radar return in decibels; their signature depends directly on the contrast between dampened capillary waves inside the slick and undampened waves in surrounding water. Option 1 models this physical damping mechanism directly by computing local background clutter. Using a moving window of 101 by 101 pixels (or 151 by 151 pixels for scenes with extensive contiguous slicks) provides a stable ocean baseline that spans typical wave fields while remaining sensitive to localized damping. SciPy uniform filtering computes this in constant time per pixel regardless of window size. High recall extraction at Stage 1 guarantees that true spills are never discarded prematurely, passing all candidate objects to Stage 2 where multi dimensional geometric and physical classifiers make the final determination.

## Follow-up

* [ ] Implement Feature 10 (Physical & Geometric Object Feature Extractor) to compute elongation, complexity, and boundary sharpness on these extracted candidates.
* [ ] Connect candidate polygons to the Feature 11 object level random forest model once trained.
