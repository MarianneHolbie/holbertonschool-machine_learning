-- trigger resets attribute valid_email if it change
CREATE TRIGGER UpdateMail
BEFORE UPDATE ON users
FOR EACH ROW
    IF NEW.email != OLD.email THEN
        SET NEW.valid_email = 0;
        END IF;
END;