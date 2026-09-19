# Rationale: 0001 Autonomous vessel and bright target masking

## Context

In synthetic aperture radar ocean surveillance, mineral oil slicks dampen surface capillary waves and appear as dark low backscatter patches. A persistent operational failure mode in radar imagery is that ship radar shadows and vessel wakes produce identical low backscatter readings in single pixel measurements. In the baseline Random Forest classifier, this ambiguity generated over 270,000 false positive pixels across the test scene.

Two physical mechanisms cause vessel false alarms. First, the metallic superstructures of vessels block radar pulses, casting a geometric shadow corridor down range behind the vessel. Second, turbulent ship wakes smooth surface capillary waves for kilometers behind moving ships. When classified purely by pixel brightness, these areas register as dark oil candidates.

The pipeline requires an autonomous solution that operates directly on any Sentinel-1 GeoTIFF without requiring external live AIS radio feeds. The solution must distinguish genuine oil discharges trailing behind a polluter from the physical radar shadow of the vessel itself.

## Options considered

### Option 1: Directional range shadow corridor with dual polarization CFAR (Recommended)

Uses vectorized local contrast filtering across VV and VH channels to detect bright corner reflectors, computes the radar range look direction, and projects geometric shadow corridors down range while applying physical damping ratio checks.

**Pros**:
- High detection accuracy on both large container ships and smaller craft.
- Directional corridor masks the actual radar void without clipping surrounding ocean indiscriminately.
- Preserves true oil spills attached to ships via physical damping validation.

**Cons**:
- Requires calculating satellite range look angle or falling back to radial dilation when orbital tags are missing.

### Option 2: Fixed radial dilation around bright reflectors

Flags pixels exceeding a static decibel threshold and applies a uniform circular dilation mask of 15 pixels around every bright point.

**Pros**:
- Very simple implementation with minimal math.
- Fast execution using basic morphological dilation.

**Cons**:
- Masks genuine oil spills that originate near a ship.
- Does not follow the elongated geometry of radar shadows down range.

### Option 3: External live AIS stream matching

Relies on real time AIS radio broadcasts to locate vessels and project shadows only where known ships report positions.

**Pros**:
- Provides vessel names, MMSI identifiers, and navigational headings directly.

**Cons**:
- Fails when ships disable AIS transponders (dark vessels).
- Fails on archived scenes or offline deployments without live internet telemetry.

## Decision

**Chosen option**: Option 1: Directional range shadow corridor with dual polarization CFAR

We implement autonomous dual polarization reflector detection with directional shadow corridor projection and physical damping verification.

## Rationale

Option 1 provides complete autonomy on any raw GeoTIFF scene. Mineral oil spills touching a ship possess physical damping ratios between $3\text{ dB}$ and $10\text{ dB}$ and sharp interfacial phase boundaries. Radar shadows, by contrast, are geometric illumination voids that lack the fluid boundary characteristics of mineral oil. Option 1 evaluates this physical contrast, preventing false suppression of true illegal discharges while eliminating shadow false positives.

## Follow-up

- Once verified in `testing/`, port `vessel_masking.py` into core `scripts/preprocessing/` during Slice 1 hardening.
- Connect detected suspect vessel positions directly into `scripts/attribution/vessel_attribution.py`.
