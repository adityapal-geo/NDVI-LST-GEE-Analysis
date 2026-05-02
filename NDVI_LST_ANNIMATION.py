# ============================================================
# CELL 1: Install & Initialize
# ============================================================
!pip install earthengine-api matplotlib imageio[ffmpeg] pillow requests -q

import ee
ee.Authenticate()
ee.Initialize(
    project='promising-idea-432505-i4',
    opt_url='https://earthengine-highvolume.googleapis.com'
)
print("✅ GEE Initialized")




# ============================================================
# CELL 2: IMPORTS & AOI
# ============================================================
import numpy as np
import requests
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter, FFMpegWriter
from IPython.display import display, Video, Image as IPImage

aoi = ee.FeatureCollection('projects/promising-idea-432505-i4/assets/durgapur')
roi = aoi.geometry()


# ============================================================
# CELL 3: FUNCTIONS
# ============================================================

# ---- Cloud Mask ----
def cloud_mask(image):
    qa = image.select('QA_PIXEL')
    mask = qa.bitwiseAnd(1 << 3).eq(0).And(qa.bitwiseAnd(1 << 4).eq(0))
    return image.updateMask(mask)

# ---- NDVI ----
def ndvi_l5(img):
    return img.normalizedDifference(['SR_B4', 'SR_B3']).rename('NDVI')

def ndvi_l8(img):
    return img.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI')

# ---- Emissivity from NDVI ----
def emissivity_from_ndvi(ndvi):
    ndvi_min = 0.2
    ndvi_max = 0.5

    pv = ((ndvi - ndvi_min) / (ndvi_max - ndvi_min)) ** 2
    pv = pv.where(ndvi.lt(ndvi_min), 0)
    pv = pv.where(ndvi.gt(ndvi_max), 1)

    return pv.multiply(0.004).add(0.986)  # ε

# ---- Brightness Temperature (Kelvin) ----
def bt_l5(img):
    return img.select('ST_B6').multiply(0.00341802).add(149.0).rename('BT')

def bt_l8(img):
    return img.select('ST_B10').multiply(0.00341802).add(149.0).rename('BT')

# ---- LST using emissivity correction formula ----
def lst_from_bt_ndvi(bt, ndvi, sensor):
    emiss = emissivity_from_ndvi(ndvi)

    # constants
    if sensor == 'L5':
        lambda_ = 11.45e-6
    else:
        lambda_ = 10.895e-6

    rho = 1.438e-2

    lst = bt.divide(
        (bt.multiply(lambda_).divide(rho).multiply(emiss.log())).add(1)
    )

    return lst.subtract(273.15).rename('LST')  # Celsius



# ============================================================
# CELL 4: NDVI + LST (ONLY LANDSAT 5 & 8) — FINAL CLEAN VERSION
# ============================================================

ndvi_palette = ['#d73027','#f46d43','#fdae61','#fee08b',
                '#d9ef8b','#a6d96a','#66bd63','#1a9850']

lst_palette = ['#2c7bb6','#00a6ca','#00ccbc',
               '#90eb9d','#ffff8c',
               '#f9d057','#f29e2e','#e76818','#d7191c']

years = list(range(2004, 2026))

ndvi_frames, lst_frames, labels, sensors = [], [], [], []

print("⬇️ Downloading frames...\n")

for year in years:
    try:
        start = f'{year}-01-01'
        end   = f'{year}-12-31'

        # ====================================================
        # SENSOR SELECTION (ONLY L5 & L8)
        # ====================================================
        if year <= 2011:
            col = (ee.ImageCollection('LANDSAT/LT05/C02/T1_L2')
                   .filterBounds(roi)
                   .filterDate(start, end)
                   .map(cloud_mask))

            ndvi_img = col.map(lambda img:
                img.normalizedDifference(['SR_B4','SR_B3']).rename('NDVI')
            ).mean().clip(roi)

            bt_img = col.map(lambda img:
                img.select('ST_B6')
                   .multiply(0.00341802)
                   .add(149.0)
                   .rename('BT')
            ).mean().clip(roi)

            lambda_ = 11.45e-6
            sensor_name = "Landsat 5"

        elif year >= 2013:
            col = (ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
                   .filterBounds(roi)
                   .filterDate(start, end)
                   .map(cloud_mask))

            ndvi_img = col.map(lambda img:
                img.normalizedDifference(['SR_B5','SR_B4']).rename('NDVI')
            ).mean().clip(roi)

            bt_img = col.map(lambda img:
                img.select('ST_B10')
                   .multiply(0.00341802)
                   .add(149.0)
                   .rename('BT')
            ).mean().clip(roi)

            lambda_ = 10.895e-6
            sensor_name = "Landsat 8"

        else:
            print(f"⚠️ {year}: skipped (no L5/L8 data)")
            continue

        # ====================================================
        # CHECK IMAGE COUNT
        # ====================================================
        if col.size().getInfo() == 0:
            print(f"⚠️ {year}: No images")
            continue

        # ====================================================
        # EMISSIVITY (SAFE GEE MATH)
        # ====================================================
        pv = ndvi_img.subtract(0.2) \
                     .divide(0.5 - 0.2) \
                     .pow(2)

        pv = pv.where(ndvi_img.lt(0.2), 0)
        pv = pv.where(ndvi_img.gt(0.5), 1)

        emiss = pv.multiply(0.004).add(0.986)

        # ====================================================
        # LST FORMULA
        # ====================================================
        rho = 1.438e-2

        lst_img = bt_img.divide(
            bt_img.multiply(lambda_)
                  .divide(rho)
                  .multiply(emiss.log())
                  .add(1)
        ).subtract(273.15).rename('LST').clip(roi)

        # ====================================================
        # VISUAL RANGES
        # ====================================================
        ndvi_stats = ndvi_img.reduceRegion(
            ee.Reducer.percentile([2,98]), roi, 30, maxPixels=1e9).getInfo()

        lst_stats = lst_img.reduceRegion(
            ee.Reducer.percentile([2,98]), roi, 30, maxPixels=1e9).getInfo()

        ndvi_min = float(ndvi_stats.get('NDVI_p2') or 0.0)
        ndvi_max = float(ndvi_stats.get('NDVI_p98') or 0.8)

        lst_min = float(lst_stats.get('LST_p2') or 20)
        lst_max = float(lst_stats.get('LST_p98') or 40)

        # ====================================================
        # DOWNLOAD NDVI
        # ====================================================
        ndvi_url = ndvi_img.getThumbURL({
            'region': roi,
            'dimensions': 512,
            'min': ndvi_min,
            'max': ndvi_max,
            'palette': ndvi_palette
        })

        ndvi_arr = np.array(Image.open(
            BytesIO(requests.get(ndvi_url).content)).convert('RGB'))

        # ====================================================
        # DOWNLOAD LST
        # ====================================================
        lst_url = lst_img.getThumbURL({
            'region': roi,
            'dimensions': 512,
            'min': lst_min,
            'max': lst_max,
            'palette': lst_palette
        })

        lst_arr = np.array(Image.open(
            BytesIO(requests.get(lst_url).content)).convert('RGB'))

        # ====================================================
        # STORE
        # ====================================================
        ndvi_frames.append(ndvi_arr)
        lst_frames.append(lst_arr)
        labels.append(str(year))
        sensors.append(sensor_name)

        print(f"✅ {year} ({sensor_name})")

    except Exception as e:
        print(f"❌ {year}: {e}")

print("\n✅ Frames ready:", len(ndvi_frames))



# ============================================================
# NDVI vs LST SCATTER + CORRELATION
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

# ---- SELECT YEAR (change if needed) ----
year = 2020

start = f'{year}-01-01'
end   = f'{year}-12-31'

# ============================================================
# LOAD IMAGE (same logic as before)
# ============================================================

if year <= 2011:
    col = (ee.ImageCollection('LANDSAT/LT05/C02/T1_L2')
           .filterBounds(roi)
           .filterDate(start, end)
           .map(cloud_mask))

    ndvi = col.map(lambda img:
        img.normalizedDifference(['SR_B4','SR_B3']).rename('NDVI')
    ).mean()

    bt = col.map(lambda img:
        img.select('ST_B6')
           .multiply(0.00341802)
           .add(149.0)
           .rename('BT')
    ).mean()

    lambda_ = 11.45e-6

else:
    col = (ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
           .filterBounds(roi)
           .filterDate(start, end)
           .map(cloud_mask))

    ndvi = col.map(lambda img:
        img.normalizedDifference(['SR_B5','SR_B4']).rename('NDVI')
    ).mean()

    bt = col.map(lambda img:
        img.select('ST_B10')
           .multiply(0.00341802)
           .add(149.0)
           .rename('BT')
    ).mean()

    lambda_ = 10.895e-6

ndvi = ndvi.clip(roi)
bt   = bt.clip(roi)

# ============================================================
# LST (same formula)
# ============================================================

pv = ndvi.subtract(0.2).divide(0.3).pow(2)
pv = pv.where(ndvi.lt(0.2), 0)
pv = pv.where(ndvi.gt(0.5), 1)

emiss = pv.multiply(0.004).add(0.986)

rho = 1.438e-2

lst = bt.divide(
    bt.multiply(lambda_).divide(rho).multiply(emiss.log()).add(1)
).subtract(273.15).rename('LST')

# ============================================================
# SAMPLE PIXELS
# ============================================================

sample = ndvi.addBands(lst).sample(
    region=roi,
    scale=30,
    numPixels=5000,
    geometries=False
).getInfo()

# Convert to numpy
ndvi_vals = np.array([f['properties']['NDVI'] for f in sample['features']])
lst_vals  = np.array([f['properties']['LST'] for f in sample['features']])

# Remove invalid
mask = np.isfinite(ndvi_vals) & np.isfinite(lst_vals)
ndvi_vals = ndvi_vals[mask]
lst_vals  = lst_vals[mask]

# ============================================================
# CORRELATION
# ============================================================

corr = np.corrcoef(ndvi_vals, lst_vals)[0,1]
print(f"📊 Correlation (r) = {corr:.3f}")

# ============================================================
# PLOT
# ============================================================

plt.figure(figsize=(7,5))

plt.scatter(ndvi_vals, lst_vals, s=5, alpha=0.4)

plt.xlabel("NDVI")
plt.ylabel("LST (°C)")
plt.title(f"NDVI vs LST ({year})\nCorrelation = {corr:.3f}")

plt.grid(True)
plt.show()





# ============================================================
# CELL 5: FINAL ANIMATION (BOTTOM-LEFT LABELS + PANEL YEAR)
# ============================================================

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors

# ---- Duration (~21 sec) ----
total_frames = len(ndvi_frames)
fps = total_frames / 21

# ---- Figure ----
fig = plt.figure(figsize=(14,7), dpi=120, facecolor='#0d1117')

# ---- Panels ----
ax1 = fig.add_axes([0.05, 0.22, 0.42, 0.65])
ax2 = fig.add_axes([0.53, 0.22, 0.42, 0.65])

ax1.axis('off')
ax2.axis('off')

im1 = ax1.imshow(ndvi_frames[0])
im2 = ax2.imshow(lst_frames[0])

# ============================================================
# YEAR (TOP-LEFT)
# ============================================================

year_txt1 = ax1.text(
    0.02, 0.97, labels[0],
    transform=ax1.transAxes,
    ha='left', va='top',
    fontsize=26, color='white', fontweight='bold',
    bbox=dict(facecolor='#000000ee', edgecolor='none', pad=6)
)

year_txt2 = ax2.text(
    0.02, 0.97, labels[0],
    transform=ax2.transAxes,
    ha='left', va='top',
    fontsize=26, color='white', fontweight='bold',
    bbox=dict(facecolor='#000000ee', edgecolor='none', pad=6)
)

# ============================================================
# SENSOR (TOP-RIGHT)
# ============================================================

sensor_txt1 = ax1.text(
    0.98, 0.97, sensors[0],
    transform=ax1.transAxes,
    ha='right', va='top',
    fontsize=10, color='#cccccc',
    bbox=dict(facecolor='#000000aa', edgecolor='none', pad=3)
)

sensor_txt2 = ax2.text(
    0.98, 0.97, sensors[0],
    transform=ax2.transAxes,
    ha='right', va='top',
    fontsize=10, color='#cccccc',
    bbox=dict(facecolor='#000000aa', edgecolor='none', pad=3)
)

# ============================================================
# PANEL LABELS (BOTTOM-LEFT)
# ============================================================

ndvi_label = ax1.text(
    0.02, 0.03, "NDVI",
    transform=ax1.transAxes,
    ha='left', va='bottom',
    fontsize=14, color='white', fontweight='bold',
    bbox=dict(facecolor='#000000cc', edgecolor='none', pad=4)
)

lst_label = ax2.text(
    0.02, 0.03, "LST (°C)",
    transform=ax2.transAxes,
    ha='left', va='bottom',
    fontsize=14, color='white', fontweight='bold',
    bbox=dict(facecolor='#000000cc', edgecolor='none', pad=4)
)

# ============================================================
# COLORBARS
# ============================================================

ndvi_palette = ['#d73027','#f46d43','#fdae61','#fee08b',
                '#d9ef8b','#a6d96a','#66bd63','#1a9850']

lst_palette = ['#2c7bb6','#00a6ca','#00ccbc',
               '#90eb9d','#ffff8c',
               '#f9d057','#f29e2e','#e76818','#d7191c']

# NDVI colorbar
cax1 = fig.add_axes([0.05, 0.12, 0.42, 0.03])
cmap1 = mcolors.LinearSegmentedColormap.from_list("ndvi", ndvi_palette)
norm1 = mcolors.Normalize(vmin=0, vmax=0.8)
cb1 = plt.colorbar(plt.cm.ScalarMappable(norm=norm1, cmap=cmap1),
                   cax=cax1, orientation='horizontal')
cb1.ax.tick_params(colors='white', labelsize=8)
cb1.set_label("NDVI", color='white', fontsize=9)

# LST colorbar
cax2 = fig.add_axes([0.53, 0.12, 0.42, 0.03])
cmap2 = mcolors.LinearSegmentedColormap.from_list("lst", lst_palette)
norm2 = mcolors.Normalize(vmin=20, vmax=45)
cb2 = plt.colorbar(plt.cm.ScalarMappable(norm=norm2, cmap=cmap2),
                   cax=cax2, orientation='horizontal')
cb2.ax.tick_params(colors='white', labelsize=8)
cb2.set_label("LST (°C)", color='white', fontsize=9)

# ============================================================
# MEANING TEXT
# ============================================================

fig.text(0.05, 0.08, "Low → High Vegetation",
         color='#66bd63', fontsize=9, fontweight='bold')

fig.text(0.75, 0.08, "Cool → Hot",
         color='#d7191c', fontsize=9, fontweight='bold')

# ============================================================
# UPDATE FUNCTION
# ============================================================

def update(i):
    im1.set_data(ndvi_frames[i])
    im2.set_data(lst_frames[i])

    year_txt1.set_text(labels[i])
    year_txt2.set_text(labels[i])

    sensor_txt1.set_text(sensors[i])
    sensor_txt2.set_text(sensors[i])

    return [im1, im2, year_txt1, year_txt2, sensor_txt1, sensor_txt2]

# ============================================================
# ANIMATION
# ============================================================

ani = animation.FuncAnimation(
    fig, update,
    frames=total_frames,
    interval=1000/fps,
    blit=True
)

plt.close()
print(f"🎬 Final animation ready (~21 sec, fps={fps:.2f})")




# ============================================================
# CELL 6: SAVE OUTPUT
# ============================================================

gif_path = '/content/NDVI_LST_formula.gif'
mp4_path = '/content/NDVI_LST_formula.mp4'

print("💾 Saving GIF...")
ani.save(gif_path, writer=PillowWriter(fps=fps))
print("✅ GIF saved")

print("💾 Saving MP4...")
ani.save(mp4_path, writer=FFMpegWriter(fps=fps, bitrate=2500))
print("✅ MP4 saved")

display(IPImage(gif_path))
display(Video(mp4_path, embed=True, width=700))