-- Trigger that decreases quantity of an item after adding a new order
CREATE TRIGGER UpdateQuantity
AFTER INSERT ON orders
FOR EACH ROW
    UPDATE items
    SET items.quantity = items.quantity - NEW.number
    WHERE items.name = NEW.item_name;
