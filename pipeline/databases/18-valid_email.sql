-- trigger resets attribute valid_email if it change
DELIMITER $$

CREATE TRIGGER UpdateMail
BEFORE UPDATE ON users
FOR EACH ROW
BEGIN
    -- if new email
    IF NEW.email != OLD.email THEN
        SET NEW.valid_email = 0;
    END IF;
END $$

DELIMITER;