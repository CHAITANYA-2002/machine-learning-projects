# Random Utilities

Welcome to the **Random Utilities** project! This project contains various Python scripts designed for diverse computational tasks, data manipulations, and experimental programming.

## 📌 Overview
The Random Utilities project includes a collection of scripts that handle multiple small but useful functionalities. Whether you need random number generation, data sampling, or string manipulations, this repository serves as a handy toolkit. These scripts can be integrated into larger projects or used as standalone utilities to automate repetitive tasks and perform quick calculations.

## 📂 Features
- **Random Number Generation**: Generate random numbers using uniform, normal, exponential, and other probability distributions.
- **Data Sampling**: Select random subsets from large datasets for efficient testing, validation, and data augmentation.
- **String Manipulations**: Convert case, remove special characters, format text, and perform advanced text processing such as tokenization and sentiment analysis.
- **Shuffling & Permutations**: Shuffle datasets, generate random sequences, and create unique permutations for experiments and cryptographic applications.
- **Mathematical Computations**: Perform quick mathematical operations, probability calculations, and statistical analysis.
- **File Handling Utilities**: Read, write, and manipulate CSV, JSON, and text files with automated formatting and validation.
- **Password & Token Generators**: Securely generate strong passwords, UUIDs, and cryptographic keys for authentication purposes.
- **Date & Time Utilities**: Generate random timestamps, calculate date differences, and format datetime objects efficiently.
- **Miscellaneous Scripts**: Includes general-purpose functions like data encryption, web scraping, and basic automation tools.

## 🚀 Running the Project
### Prerequisites
Ensure you have Python installed along with the necessary libraries:
```bash
pip install numpy pandas random datetime uuid
```

### Steps to Run
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/CHAITANYA-2002/Machine-Learning.git
   ```
2. **Navigate to the project folder**:
   ```bash
   cd random
   ```
3. **Run a specific script**:
   ```bash
   python random_script.py
   ```

### Example Usage
#### Random Number Generation
```python
import random
print(random.randint(1, 100))  # Generates a random integer between 1 and 100
```

#### Data Sampling
```python
import random
sample_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
print(random.sample(sample_data, 3))  # Returns 3 random elements from the list
```

#### String Manipulation
```python
def clean_text(text):
    return text.strip().lower().replace(" ", "_")
print(clean_text("  Hello World  "))  # Output: hello_world
```

#### Password Generation
```python
import random
import string
def generate_password(length=12):
    characters = string.ascii_letters + string.digits + string.punctuation
    return ''.join(random.choice(characters) for _ in range(length))
print(generate_password())  # Outputs a random secure password
```

#### Date & Time Utility
```python
from datetime import datetime, timedelta
random_date = datetime.now() - timedelta(days=random.randint(1, 365))
print(random_date.strftime('%Y-%m-%d %H:%M:%S'))  # Outputs a random past date
```

## 📊 Use Cases
- Generating random datasets for testing machine learning models and simulations.
- Creating randomized selections from large datasets for unbiased data analysis.
- Shuffling and organizing datasets efficiently for research and data science experiments.
- Formatting and cleaning text data for preprocessing NLP models.
- Automating password generation and security key creation.
- Generating synthetic timestamps and event data for software testing.
- Quick utility functions for automation scripts in data processing and system operations.

## 🛠 Contributions
Contributions are welcome! If you have improvements, bug fixes, or new utility scripts, feel free to submit a pull request. 

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
### 📬 Contact
For any queries, feel free to reach out via GitHub Issues.

Explore Randomness with Python! 🎲🚀

