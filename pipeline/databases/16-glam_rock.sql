-- old school band
SELECT band_name,  IF NULL(split, 2020) - formed AS lifespan
FROM metal_bands
WHERE style LIKE '%Glam rock%'
ORDER BY lifespan DESC;