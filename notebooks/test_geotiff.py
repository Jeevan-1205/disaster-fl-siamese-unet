from src.xbd_geotiff_dataset import XBDGeoDataset

ds = XBDGeoDataset(
"data/xbd_full/geotiffs/tier1"
)

print(
len(ds)
)

pre,post,mask=ds[0]

print(
pre.shape
)

print(
post.shape
)

print(
mask.shape
)