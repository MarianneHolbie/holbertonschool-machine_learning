-- Function that divides (and return) two numbers
DELIMITER $$

CREATE FUNCTION SafeDiv(
    IN a INT,
    IN b INT)
    RETURNS FLOAT
    DETERMINISTIC

    BEGIN
        IF b = 0
            BEGIN
                RETURN 0;
            END
        ELSE
            BEGIN
                RETURN a / b;
            END
        END IF;
    END $$

DELIMITER;
