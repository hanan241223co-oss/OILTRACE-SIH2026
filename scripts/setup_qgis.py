from qgis.core import (
    QgsProject,
    QgsRasterLayer,
    QgsVectorLayer,
    QgsFillSymbol,
    QgsLineSymbol,
    QgsSingleBandGrayRenderer
)

import os

project = QgsProject.instance()

# ============================================================
# OILTRACE QGIS INVESTIGATION VIEW
# ============================================================

SCENE = "00000"


# ============================================================
# HELPER: REMOVE EXISTING LAYERS
# ============================================================

def remove_layer(name):
    layers = project.mapLayersByName(name)

    for layer in layers:
        project.removeMapLayer(layer.id())


# Remove old copies before loading new ones
remove_layer("OpenStreetMap")
remove_layer("Ocean / Water Body")
remove_layer("SAR Scene 00000")
remove_layer("Detected Oil Spill")
remove_layer("Drift Hindcast (12h)")
remove_layer("Drift Forecast (12h)")


# ============================================================
# 1. OPENSTREETMAP BASE MAP
# ============================================================

osm_url = (
    "type=xyz&url=https://tile.openstreetmap.org/"
    "{z}/{x}/{y}.png&zmax=19&zmin=0&crs=EPSG3857"
)

osm = QgsRasterLayer(
    osm_url,
    "OpenStreetMap",
    "wms"
)

if osm.isValid():
    project.addMapLayer(osm)
    print("OpenStreetMap: Loaded")
else:
    print("WARNING: OpenStreetMap could not be loaded.")


# ============================================================
# 2. WATER BODY
# ============================================================

water_path = (
    r"D:\OILTRACE\data\gis\water\ne_10m_ocean.shp"
)

water = QgsVectorLayer(
    water_path,
    "Ocean / Water Body",
    "ogr"
)

if water.isValid():

    project.addMapLayer(water)

    water_symbol = QgsFillSymbol.createSimple({
        "color": "0,0,0,0",
        "outline_color": "70,150,200,180",
        "outline_width": "0.3"
    })

    water.renderer().setSymbol(water_symbol)
    water.triggerRepaint()

    print("Water Body: Loaded")

else:
    print("WARNING: Water layer could not be loaded.")


# ============================================================
# 3. SAR SCENE
# ============================================================

sar_path = (
    r"D:\OILTRACE\data\images\oil_spill\Oil\00000.tif"
)

if os.path.exists(sar_path):

    sar = QgsRasterLayer(
        sar_path,
        "SAR Scene 00000"
    )

    if sar.isValid():
        project.addMapLayer(sar)
        print("SAR Scene 00000: Loaded")
    else:
        print("WARNING: SAR Scene could not be loaded.")

else:
    print("WARNING: SAR file not found.")


# # ============================================================
# 4. DETECTED OIL SPILL
# ============================================================

spill_path = (
    r"D:\OILTRACE\data\results\detection\00000_spills.geojson"
)

if os.path.exists(spill_path):

    spill = QgsVectorLayer(
        spill_path,
        "Detected Oil Spill",
        "ogr"
    )

    if spill.isValid():

        project.addMapLayer(spill)

        # All detected components:
        # transparent fill + visible red outline
        spill_symbol = QgsFillSymbol.createSimple({
            "color": "255,59,48,20",
            "outline_color": "255,59,48,220",
            "outline_width": "0.5"
        })

        spill.renderer().setSymbol(spill_symbol)
        spill.triggerRepaint()

        print("Detected Oil Spill: Loaded and Styled")

    else:
        print("WARNING: Spill layer is invalid.")

else:
    print("WARNING: 00000_spills.geojson was not found.")

# ============================================================
# 5. DRIFT HINDCAST
# ============================================================

hindcast_path = (
    r"D:\OILTRACE\data\results\drift\spill_00000_hindcast.geojson"
)

if os.path.exists(hindcast_path):

    hindcast = QgsVectorLayer(
        hindcast_path,
        "Drift Hindcast (12h)",
        "ogr"
    )

    if hindcast.isValid():

        project.addMapLayer(hindcast)

        hindcast_symbol = QgsLineSymbol.createSimple({
            "line_color": "255,214,10,255",
            "line_width": "1.2"
        })

        hindcast.renderer().setSymbol(hindcast_symbol)
        hindcast.triggerRepaint()

        print("Drift Hindcast: Loaded")

    else:
        print("WARNING: Hindcast layer is invalid.")

else:
    print("WARNING: Hindcast file not found.")


# ============================================================
# 6. DRIFT FORECAST
# ============================================================

forecast_path = (
    r"D:\OILTRACE\data\results\drift\spill_00000_forecast.geojson"
)

if os.path.exists(forecast_path):

    forecast = QgsVectorLayer(
        forecast_path,
        "Drift Forecast (12h)",
        "ogr"
    )

    if forecast.isValid():

        project.addMapLayer(forecast)

        forecast_symbol = QgsLineSymbol.createSimple({
            "line_color": "0,122,255,255",
            "line_width": "1.2"
        })

        forecast.renderer().setSymbol(forecast_symbol)
        forecast.triggerRepaint()

        print("Drift Forecast: Loaded")

    else:
        print("WARNING: Forecast layer is invalid.")

else:
    print("WARNING: Forecast file not found.")


# ============================================================
# 7. CREATE LAYER GROUPS
# ============================================================

root = project.layerTreeRoot()

for group_name in [
    "01 BASE MAP",
    "02 CONTEXT",
    "03 OILTRACE ANALYSIS"
]:

    old_group = root.findGroup(group_name)

    if old_group:
        root.removeChildNode(old_group)


base_group = root.insertGroup(
    0,
    "01 BASE MAP"
)

context_group = root.insertGroup(
    1,
    "02 CONTEXT"
)

analysis_group = root.insertGroup(
    2,
    "03 OILTRACE ANALYSIS"
)


# ============================================================
# 8. MOVE LAYERS INTO GROUPS
# ============================================================

def move_to_group(layer_name, group):

    layers = project.mapLayersByName(layer_name)

    if not layers:
        return

    layer = layers[0]

    node = root.findLayer(layer.id())

    if node:

        clone = node.clone()

        group.addChildNode(clone)

        root.removeChildNode(node)


move_to_group(
    "OpenStreetMap",
    base_group
)

move_to_group(
    "Ocean / Water Body",
    context_group
)

move_to_group(
    "SAR Scene 00000",
    analysis_group
)

move_to_group(
    "Detected Oil Spill",
    analysis_group
)

move_to_group(
    "Drift Hindcast (12h)",
    analysis_group
)

move_to_group(
    "Drift Forecast (12h)",
    analysis_group
)


# ============================================================
# 9. SAVE PROJECT
# ============================================================

project.write(
    r"D:\OILTRACE\qgis\OILTRACE.qgz"
)


# ============================================================
# DONE
# ============================================================

print("")
print("========================================")
print("OILTRACE INVESTIGATION VIEW COMPLETE")
print("========================================")
print("Scene: 00000")
print("OpenStreetMap: Loaded")
print("Water Body: Loaded")
print("SAR: Loaded")
print("Oil Spill: Loaded + Styled")
print("Hindcast: Loaded")
print("Forecast: Loaded")
print("Project: Saved")
print("========================================")

# Display SAR as clean grayscale
if sar.isValid():
    sar.setRenderer(
        QgsSingleBandGrayRenderer(
            sar.dataProvider(),
            1
        )
    )
    sar.triggerRepaint()

    # ============================================================
# 5. PRIMARY DETECTION COMPONENT
# ============================================================

from qgis.core import (
    QgsFeature,
    QgsVectorLayer,
    QgsFillSymbol
)

# Find the largest detected component
if spill.isValid() and spill.featureCount() > 0:

    primary_feature = max(
        spill.getFeatures(),
        key=lambda f: f["pixel_count"]
    )

    # Create temporary in-memory layer
    primary = QgsVectorLayer(
        "Polygon?crs=EPSG:4326",
        "Primary Detection",
        "memory"
    )

    provider = primary.dataProvider()

    # Copy original fields
    provider.addAttributes(spill.fields())
    primary.updateFields()

    # Copy largest feature
    new_feature = QgsFeature(primary.fields())
    new_feature.setGeometry(primary_feature.geometry())
    new_feature.setAttributes(primary_feature.attributes())

    provider.addFeature(new_feature)
    primary.updateExtents()

    # Add to project
    project.addMapLayer(primary)

    # Prominent primary spill styling
    primary_symbol = QgsFillSymbol.createSimple({
        "color": "255,59,48,80",
        "outline_color": "255,59,48,255",
        "outline_width": "1.2"
    })

    primary.renderer().setSymbol(primary_symbol)
    primary.triggerRepaint()

    print("")
    print("PRIMARY DETECTION CREATED")
    print(
        "Component ID:",
        primary_feature["component_id"]
    )
    print(
        "Pixel Count:",
        primary_feature["pixel_count"]
    )
    print(
        "Area (km2):",
        primary_feature["area_km2"]
    )

else:
    print("WARNING: No spill features available.")


# ============================================================
# 6. SAVE PROJECT
# ============================================================

project.write(
    r"D:\OILTRACE\qgis\OILTRACE.qgz"
)

print("Primary Detection: Loaded and Styled")
print("Project: Saved")