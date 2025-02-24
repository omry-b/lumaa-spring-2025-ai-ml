# Content-Based Recipe Recommendation System

A simple content-based recommendation system that suggests recipes based on text descriptions of what you're looking for. Built as part of the AI/ML Intern Challenge.

## Dataset

This project uses a curated recipe dataset containing approximately 500 recipes with titles and descriptions. The dataset has been cleaned and preprocessed to remove duplicates and ensure quality recommendations.

### Dataset Structure
- `title`: Name of the recipe
- `description`: Detailed description of the recipe, including ingredients and preparation methods

## Setup

### Prerequisites
- Python 3.8 or higher
- Virtual environment (recommended)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/omry-b/recipe-recommender.git
cd recipe-recommender
```

2. Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip3 install -r requirements.txt
```

## Running the System

1. From the command line:
```bash
python3 recommend.py
```

2. When prompted, enter a description of what you're looking for. For example:
   - "A spicy vegetarian dish with lots of vegetables"
   - "Something sweet with chocolate and berries"
   - "A quick and easy pasta dish with tomato sauce"

### Example Output

```
Welcome to the Content-Based Recommendation System!
------------------------------------------------

Please describe what you're looking for: I want a healthy salad with quinoa

Top Recommendations:
-------------------

1. Mediterranean Quinoa Salad
   Similarity Score: 0.8754
   Description: A refreshing salad featuring quinoa, cherry tomatoes, cucumber, and feta cheese, dressed with olive oil and lemon juice...

2. Quinoa Power Bowl
   Similarity Score: 0.7652
   Description: Protein-rich quinoa bowl with roasted vegetables, chickpeas, and avocado, topped with a tahini dressing...

[Additional recommendations...]
```

## Implementation Details

The recommendation system uses:
- TF-IDF (Term Frequency-Inverse Document Frequency) for text vectorization
- Cosine similarity for finding matching recipes
- Bigram support for better phrase matching
- Efficient feature limiting for quick responses

## Salary Expectation

Expected monthly salary: $4,000 - $5,000 USD

## Future Improvements

Potential enhancements that could be made with more time:
- Add support for dietary restrictions filtering
- Implement ingredient-based filtering
- Add recipe ratings and cooking time information
- Enhance the similarity algorithm with word embeddings
- Add a web interface for easier interaction

## Contact

Omry Bejerano
- Email: omry@stanford.edu
- GitHub: https://github.com/omry-b
- LinkedIn: https://www.linkedin.com/in/omry-bejerano/
