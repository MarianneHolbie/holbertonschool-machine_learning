-- import database and displays average temperature by city order by temp (desc)
SELECT city, AVG(value) AS avg_temp
FROM temperatures
GROUP BY city
ORDER by avg_temp DESC;