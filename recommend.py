"""
Content-Based Recommendation System
---------------------------------
This food recom,endation module implements a simple content-based recommendation system 
using TF-IDF and cosine similarity. It processes text descriptions to suggest similar items 
based on content similarity.

Author: Omry Bejerano (Stanford University)
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ContentRecommender:
    # A content-based recommendation system using TF-IDF and cosine similarity
    
    def __init__(self, data_path: str):
        # Initialize the recommender system
        self.data_path = data_path
        self.df = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        
    def load_data(self) -> bool:
        # Load & process the data (return true if successful, false otherwise)
        try:
            self.df = pd.read_csv(self.data_path)
            
            # Basic data cleaning (drop header & duplicates)
            self.df = self.df.dropna(subset=['title', 'description'])
            self.df = self.df.drop_duplicates(subset=['title', 'description'])
            
            logger.info(f"Loaded dataset with {len(self.df)} items")
            return True
            
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            return False
    
    def build_vectors(self) -> bool:
        # Build TF-IDF vectors from the item descriptions (true is vectorization was successful, false otherwise)
        try:
            self.tfidf_vectorizer = TfidfVectorizer(
                stop_words='english',
                max_features=5000,  # Limit features for efficiency
                ngram_range=(1, 2)  # Include bigrams for better matching
            )
            
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(
                self.df['description'].fillna('')
            )
            
            logger.info(f"Built TF-IDF matrix with shape {self.tfidf_matrix.shape}")
            return True
            
        except Exception as e:
            logger.error(f"Error building vectors: {e}")
            return False
    
    def get_recommendations(
        self, 
        query: str, 
        top_n: int = 5
    ) -> List[Tuple[str, str, float]]:
        # Get recommendations based on a text query
        try:
            # Vectorize the query
            query_vec = self.tfidf_vectorizer.transform([query])
            
            # Compute similarities
            similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
            
            # Get top N recommendations
            top_indices = similarities.argsort()[::-1][:top_n]
            
            recommendations = [
                (
                    self.df.iloc[idx]['title'],
                    self.df.iloc[idx]['description'],
                    float(similarities[idx])
                )
                for idx in top_indices
            ]
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error getting recommendations: {e}")
            return []

def main():
    try:
        # Initialize recommender
        recommender = ContentRecommender('cleaned_recipes.csv')
        
        # Load and prepare data
        if not recommender.load_data():
            logger.error("Failed to load data. Exiting.")
            return
            
        if not recommender.build_vectors():
            logger.error("Failed to build vectors. Exiting.")
            return
        
        # Get user input
        print("\nWelcome to the Content-Based Recommendation System!")
        print("------------------------------------------------")
        query = input("\nPlease describe what you're looking for: ").strip()
        
        if not query:
            print("Error: Description cannot be empty.")
            return
            
        # Get and display recommendations
        recommendations = recommender.get_recommendations(query)
        
        if not recommendations:
            print("\nNo recommendations found. Please try a different description.")
            return
            
        print("\nTop Recommendations:")
        print("-------------------")
        for i, (title, desc, score) in enumerate(recommendations, 1):
            print(f"\n{i}. {title}")
            print(f"   Similarity Score: {score:.4f}")
            print(f"   Description: {desc[:200]}...")

    except KeyboardInterrupt:
        print("\nProgram terminated by user.")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")

if __name__ == "__main__":
    main()
