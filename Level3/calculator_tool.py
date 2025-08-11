def calculate(expression: str) -> float:
    allowed_chars = set("0123456789+-*/. ")
    if not all(c in allowed_chars for c in expression):
        raise ValueError("Invalid characters in expression.")
    try:
        return eval(expression)
    except Exception as e:
        raise ValueError(f"Error in calculation: {str(e)}")
