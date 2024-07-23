-- old school band
SELECT band_name, IF (split is NULL, YEAR(CURRENT_DATE()), split) - formed AS lifespan
FROM metal_bands
WHERE style LIKE '%Glam rock%'
ORDER BY lifespan DESC;