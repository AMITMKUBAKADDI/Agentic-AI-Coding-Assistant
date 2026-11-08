"""Main application runner demonstrating calculator functions."""

from calculator import add, divide

def run():
    print("Welcome to the Sample Project!")
    result = add(10, 5)
    print(f"10 + 5 = {result}")
    
    div_result = divide(20, 4)
    print(f"20 / 4 = {div_result}")

if __name__ == "__main__":
    run()

