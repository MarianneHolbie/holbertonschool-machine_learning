-- lists all genres by their rating
SELECT tg.name, SUM(tsr.rate) AS rating
FROM tv_genre AS tg
JOIN tv_show_genres AS tsg
ON tsr.show_id = ts.id
GROUP BY tsg.genre_id = tg.id
JOIN tsr
ON tsr.show_id = tsg.show_id
GROUP BY tg.name
ORDER BY rating DESC;