def apply_tax(amount, percent):
    """Adds tax percentage to amount and returns rounded value."""
    return round(amount + amount * percent / 100, 2)