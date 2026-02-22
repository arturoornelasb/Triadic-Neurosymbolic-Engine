import pandas as pd

def generate_catalog():
    data = [
        {"id": 101, "name": "Hyper Speed Red Sports Car", "category": "Vehicles"},
        {"id": 102, "name": "Heavy Duty Yellow Dump Truck", "category": "Vehicles"},
        {"id": 103, "name": "Electric City Bicycle", "category": "Vehicles"},
        {"id": 104, "name": "Fresh Organic Red Apples", "category": "Food"},
        {"id": 105, "name": "Yellow Chiquita Bananas", "category": "Food"},
        {"id": 106, "name": "Whole Wheat Sliced Bread", "category": "Food"},
        {"id": 107, "name": "Queen's Golden Crown Replica", "category": "Accessories"},
        {"id": 108, "name": "King's Velvet Red Cape", "category": "Apparel"},
        {"id": 109, "name": "Fast Silver Racing Motorcycle", "category": "Vehicles"},
        {"id": 110, "name": "Juicy Ripe Oranges Pack", "category": "Food"},
    ]
    
    df = pd.DataFrame(data)
    df.to_csv("tests/sample_catalog.csv", index=False)
    print("Created synthetic catalog at 'tests/sample_catalog.csv' with 10 records.")

if __name__ == "__main__":
    generate_catalog()
