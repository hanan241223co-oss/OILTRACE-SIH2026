# 0001. Autonomous vessel and bright target masking

**Date**: 2026-09-19
**Status**: Accepted

## Summary

This specification establishes an autonomous vessel reflector detector and directional radar shadow corridor masking module for dual polarization Sentinel-1 synthetic aperture radar scenes. Metal ships, oil platforms, and marine structures act as strong corner reflectors that create high radar backscatter followed by dark shadow corridors in the line of sight direction. By identifying these bright point reflectors and masking their projected shadow geometry, the detection pipeline eliminates false alarm oil slicks caused by vessels while preserving true oil discharges through physical damping contrast checks.

## Requirements

**User stories**:
- As a maritime surveillance operator, I want the system to detect vessels directly from radar backscatter so that vessel radar shadows and wake corridors do not trigger false spill alarms.
- As an environmental investigator, I want genuine oil spills originating from discharging ships to be preserved so that actual illegal discharges are never masked out.

**Acceptance criteria**:
- **AC-1**: Identify metallic vessel reflectors using vectorized dual polarization peak detection ($VV > -2.0\text{ dB}$ or $VH > -12.0\text{ dB}$ with local contrast exceeding $10\text{ dB}$ above the $51 \times 51$ moving window local mean), recording pixel centroid, geographic coordinates, and peak backscatter.
- **AC-2**: Construct a directional radar shadow corridor projecting down range from each detected vessel up to 25 pixels (width equal to vessel reflector width plus 4 pixels padding), falling back to a 15-pixel radial dilation buffer when satellite orbit azimuth metadata is absent.
- **AC-3**: Suppress candidate dark spot polygons that fall within the shadow corridor when their local damping contrast is below $2.5\text{ dB}$.
- **AC-4**: Preserve candidate oil slicks touching a vessel when physical damping exceeds $2.5\text{ dB}$ and boundary Sobel gradient mean exceeds $12.0$, marking the vessel entity as a suspect spill source (`is_suspect_source = True`).
- **AC-5**: Export detected vessel entities as a GeoJSON FeatureCollection of Point features alongside spill vector outputs for downstream attribution.

## Feature design

**Data model sketch**:

```
┌───────────────────────────────────────────────────────────┐
│ VesselTarget                                              │
├───────────────────────────────────────────────────────────┤
│ target_id: str (Primary Key, e.g. "vessel_001")          │
│ pixel_x: float (Raster column centroid)                   │
│ pixel_y: float (Raster row centroid)                      │
│ longitude: float (WGS84 geographic coordinate)            │
│ latitude: float (WGS84 geographic coordinate)             │
│ peak_vv_db: float (Maximum VV backscatter in decibels)    │
│ peak_vh_db: float (Maximum VH backscatter in decibels)    │
│ pixel_area: int (Reflector footprint size in pixels)      │
│ is_suspect_source: bool (True if touching true slick)     │
└───────────────────────────────────────────────────────────┘
                             │
                             │ 1:1
                             ▼
┌───────────────────────────────────────────────────────────┐
│ ShadowCorridor                                            │
├───────────────────────────────────────────────────────────┤
│ corridor_id: str (Primary Key)                            │
│ target_id: str (Foreign Key -> VesselTarget)              │
│ geometry_type: str ("wedge" or "radial_buffer")           │
│ range_azimuth_deg: float (Look angle in degrees)          │
│ length_pixels: float (Projection length down range)       │
│ width_pixels: float (Cross range width)                   │
│ suppressed_pixel_count: int (Count of masked candidates)  │
└───────────────────────────────────────────────────────────┘
```

**State transitions**:
Not applicable. Vessel targets are immutable extraction records generated per satellite scene.

**API surface**:

| Function | Module | Key inputs | Key outputs | Auth | Key errors |
|---|---|---|---|---|---|
| `detect_vessel_targets` | `testing/vessel_masking.py` | `vv_db: ndarray`, `vh_db: ndarray`, `valid_mask: ndarray`, `metadata: dict` | `list[VesselTarget]` | None | Invalid array shape, missing bands |
| `project_shadow_corridors` | `testing/vessel_masking.py` | `vessels: list[VesselTarget]`, `shape: tuple`, `candidate_mask: ndarray`, `metadata: dict` | `shadow_mask: ndarray`, `corridors: list[ShadowCorridor]` | None | Metadata CRS projection failure |
| `mask_vessel_shadows` | `testing/vessel_masking.py` | `sar_data: ndarray`, `candidate_mask: ndarray`, `metadata: dict`, `output_geojson_path: str = None` | `cleaned_mask: ndarray`, `vessels: list[VesselTarget]`, `geojson: dict` | None | Non positive resolution, empty scene |

**Value sourcing**:

| Action | Value produced / displayed | Source |
|---|---|---|
| Reflector detection | `peak_vv_db`, `peak_vh_db` | Maximum intensity in raw Sentinel-1 float32 bands |
| Coordinate transform | `latitude`, `longitude` | Derived from raster `pixel_x`, `pixel_y` using `rasterio.transform.xy` and `metadata['transform']` |
| Shadow geometry | `range_azimuth_deg` | Inferred from `metadata.get('look_direction')` or fallback 15-pixel radial dilation |
| Shadow suppression | `cleaned_mask` | Logical difference between candidate mask and unverified shadow pixels |
| Vessel export | GeoJSON FeatureCollection | Assembled Point features saved to optional `output_geojson_path` |

**Key invariants**:
- Vessel detection runs only over valid ocean pixels, excluding nodata swath borders.
- Dense clutter cap: maximum 300 vessel targets per scene, sorted by `peak_vv_db` in descending order to retain the strongest reflectors and prevent port clutter explosions.
- Preservation override: candidate slicks with damping ratio $DR > 2.5\text{ dB}$, boundary Sobel gradient mean $\ge 12.0$, and area $\ge 50$ pixels are preserved even when touching shadow corridors.

**Security model**:
Local file system operations only. No remote network calls, no stored credentials, and no external attack surface.

**Critical test scenarios**:
- Happy path: Vessel detected with $VV > 0\text{ dB}$ and shadow corridor suppresses adjacent false alarm speck, verifies AC-1, AC-2, AC-3.
- Discharging vessel preservation: Vessel with trailing slick ($DR = 3.2\text{ dB}$, edge gradient 14.5) preserves the slick and flags the vessel as `is_suspect_source = True`, verifies AC-4.
- Missing georeferencing fallback: Scene without flight azimuth falls back gracefully to 15-pixel radial buffer, verifies AC-2.
- Clean ocean negative scene: Scene with zero vessels returns empty vessel list and unmodified candidate mask, verifies AC-1, AC-5.

## Build plan (Tracer Bullet approach)

- [x] **Step 1 (Core Thread)**: Implement vectorized dual polarization vessel reflector detection in `testing/vessel_masking.py`, converting pixel coordinates to geographic coordinates via metadata transform, satisfying AC-1.
- [x] **Step 2 (Corridor Projection)**: Implement directional shadow corridor projection and fallback radial dilation with suppressed pixel tallying, satisfying AC-2.
- [x] **Step 3 (Contrast Preservation)**: Implement physical damping contrast check and slick preservation override, satisfying AC-3 and AC-4.
- [x] **Step 4 (Geospatial Export & Integration)**: Add GeoJSON Point export and integrate `mask_vessel_shadows` into `testing/detect_oil.py`, satisfying AC-5.
- [x] **Step 5 (Scene Verification)**: Run end to end verification against test scene `testing/det_20260917163850_004aae/scene.tif`, proving vessel shadow suppression without loss of true oil pixels.

## Consequences

- False alarm dark spots adjacent to ships are eliminated before object classification.
- Suspect vessels are formally recorded with geographic positions for downstream AIS attribution.
- Slightly higher computation time during preprocessing (roughly $0.15$ seconds per scene).
