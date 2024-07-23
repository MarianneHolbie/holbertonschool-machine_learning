-- lists all genres by their rating
SELECT tg.name, SUM(tsr.rate) AS rating
FROM tv_genres AS tg
JOIN tv_show_genres AS tsg
ON tsg.genre_id = tg.id
JOIN tv_show_ratings AS tsr
ON tsr.show_id = tsg.show_id
GROUP BY tg.name
ORDER BY rating DESC;