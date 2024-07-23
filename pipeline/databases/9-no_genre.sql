-- No genre
SELECT ts.title, tsg.genre_id
FROM tv_shows AS ts
JOIN tv_show_genres AS tsg
ON ts.id = tsg.show_id
WHERE tsg.genre_id = NULL
ORDER BY ts.title ASC, tsg.genre_id ASC;