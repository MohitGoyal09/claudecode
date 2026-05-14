import sys

def main():
    print("=" * 30)
    print("   Welcome to Hello World!")
    print("=" * 30)
    name = input("Enter your name: ").strip()
    
    if not name:
        name = "World"
    
    print()
    print(f"Hello, {name}!")
    print("Have a great day!")
    print("=" * 30)

if __name__ == "__main__":
    main()