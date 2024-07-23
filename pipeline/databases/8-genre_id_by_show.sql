-- lists all shows that have one genre linked
SELECT ts.title, tsg.genre_id
FROM tv_show AS ts
JOIN tv_show_genres AS tsg
ON ts.id = tsg.show_id
ORDER BY ts.title ASC, tsg.genre_id;