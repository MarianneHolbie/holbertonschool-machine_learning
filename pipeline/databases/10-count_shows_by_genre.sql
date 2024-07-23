-- number of shows by genre
SELECT tg.name AS genre, COUNT(tsg.genre_id) AS number_of_shows
FROM tv_show_genres AS tsg
JOIN tv_genres AS tg
ON tg.id = tsg.genre_id
GROUP BY genre
ORDER BY number_of_shows DESC;