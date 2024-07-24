-- Function that divides (and return) two numbers
DELIMITER $$

CREATE FUNCTION SafeDiv(
    a INT,
    b INT)
    RETURNS FLOAT
    DETERMINISTIC

    BEGIN
        IF b = 0 THEN
            RETURN 0;
        ELSE
            RETURN a / b;
    END $$

DELIMITER;
