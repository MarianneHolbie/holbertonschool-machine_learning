-- stored procedure AddBonus that adds a new correction for a student
DELIMITER $$

-- create procedure with params
CREATE PROCEDURE AddBonus(
    user_id INT,
    project_name VARCHAR(255),
    score INT)

BEGIN
    -- creation or update projects
    IF NOT EXISTS (
        SELECT name
        FROM projects
        WHERE name = project_name
        )
        THEN
        INSERT INTO projects(name)
        VALUES (project_name);
    END IF;

    -- create corresponding corrections row
    INSERT INTO corrections (
        user_id,
        project_id
        score)
        VALUES (
        user_id (SELECT id FROM projects
                    WHERE name = project_name),
        score);

END $$
DELIMITER;



END $$

DELIMITER;
