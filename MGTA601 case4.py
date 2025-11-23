import pandas as pd
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 1. Data Loading and Preprocessing
# Purpose: Load, clean, and restructure data to fit recommendation system requirements
# Reference: PPT highlights recommendation data characteristics (purchase/ratings matrices)

def load_and_preprocess_data():
   
    # Load data from the specified CSV file
    df = pd.read_csv('F:\桌面\Python Coding\data sources\BigBasket.csv')

    # Clean data: Remove duplicates and standardize product names
    df['Description'] = df['Description'].str.strip().str.lower()
    df = df.drop_duplicates(subset=['Member', 'Description'])  # One purchase record per user - product

    # Create user - item purchase matrix
    user_item_matrix = df.pivot_table(
        index='Member',
        columns='Description',
        values='Order',  # Use Order presence as purchase indicator
        aggfunc='count',
        fill_value=0
    )
    # Convert to binary matrix (1 = purchased, 0 = not purchased)
    user_item_matrix = (user_item_matrix > 0).astype(int)

    # Check if the generated user - item matrix is empty
    if user_item_matrix.empty:
        print("The generated user - item matrix is empty. Please check the data content and processing logic.")
        return None, None

    return df, user_item_matrix

# Execute data preprocessing
df, user_item_matrix = load_and_preprocess_data()
if df is None or user_item_matrix is None:
    print("Data preprocessing failed. Exiting.")
else:
    print(f"Data preprocessing completed. User - item matrix shape: {user_item_matrix.shape}")
    print("User - item purchase matrix:\n", user_item_matrix.head())

# Proceed only if data preprocessing was successful
if user_item_matrix is not None:

    # 2. Similarity Calculation (Foundational for K - NN and Collaborative Filtering)
    # Purpose: Compute user - user and item - item similarity using cosine similarity (Similarity Measures)
   
    def calculate_similarity(matrix, similarity_type='user'):
        """
        Calculate similarity matrix using cosine similarity (PPT: Similarity for Classification/Recommendations)
        - similarity_type: 'user' (user - user similarity) or 'item' (item - item similarity)
        """
        try:
            if similarity_type == 'user':
                # User - user similarity: Rows = users, Columns = users
                similarity_matrix = cosine_similarity(matrix)
                similarity_df = pd.DataFrame(
                    similarity_matrix,
                    index=matrix.index,
                    columns=matrix.index
                )
            elif similarity_type == 'item':
                # Item - item similarity: Rows = items, Columns = items
                similarity_matrix = cosine_similarity(matrix.T)
                similarity_df = pd.DataFrame(
                    similarity_matrix,
                    index=matrix.columns,
                    columns=matrix.columns
                )
            else:
                raise ValueError("Similarity type must be 'user' or 'item'")
            return similarity_df
        except Exception as e:
            print(f"Error calculating similarity matrix: {e}")
            return None


    # Calculate user - user and item - item similarity matrices
    user_similarity = calculate_similarity(user_item_matrix, similarity_type='user')
    item_similarity = calculate_similarity(user_item_matrix, similarity_type='item')

    if user_similarity is not None and item_similarity is not None:
        print("\nUser - user similarity matrix (top 3x3):\n", user_similarity.iloc[:3, :3].round(2))
        print("\nItem - item similarity matrix (top 3x3):\n", item_similarity.iloc[:3, :3].round(2))
    else:
        print("Failed to calculate similarity matrices.")

  
    # 3. Collaborative Filtering Recommendation Algorithms 
    # Implements: User - Based CF and Item - Based CF (Collaborative Filtering)
   
    def user_based_collaborative_filtering(
            user_id,
            user_item_matrix,
            user_similarity,
            k=3,  # Number of nearest neighbors (K - NN Classification)
            top_n=5  # Number of recommendations
    ):
        """
        User - Based Collaborative Filtering (User Based Collaborative Filtering)
        Logic: Recommend items liked by similar users that the target user hasn't purchased
        """
        try:
            # Step 1: Get k most similar users (exclude self)
            similar_users = user_similarity[user_id].drop(user_id).sort_values(ascending=False).head(k)

            # Step 2: Get items purchased by similar users but not by target user
            target_user_purchases = user_item_matrix.loc[user_id]
            recommended_items = []

            for similar_user, similarity_score in similar_users.items():
                # Items purchased by similar user
                similar_user_purchases = user_item_matrix.loc[similar_user]
                # Items not purchased by target user
                new_items = similar_user_purchases[target_user_purchases == 0].index.tolist()
                # Weight items by user similarity (PPT: Weighted Voting)
                recommended_items.extend([(item, similarity_score) for item in new_items])

            # Step 3: Rank items by total similarity weight and deduplicate
            item_scores = defaultdict(float)
            for item, score in recommended_items:
                item_scores[item] += score

            # Sort items by score (highest first) and select top N
            sorted_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)
            top_recommendations = [item for item, score in sorted_items[:top_n]]

            return top_recommendations, similar_users
        except Exception as e:
            print(f"Error in user - based collaborative filtering: {e}")
            return [], None


    def item_based_collaborative_filtering(
            user_id,
            user_item_matrix,
            item_similarity,
            k=3,  # Number of similar items 
            top_n=5  # Number of recommendations
    ):
        """
        Item - Based Collaborative Filtering (Item Based Collaborative Filtering)
        Logic: Recommend items similar to those the target user has purchased
        """
        try:
            # Step 1: Get items purchased by target user
            target_user_purchases = user_item_matrix.loc[user_id]
            purchased_items = target_user_purchases[target_user_purchases == 1].index.tolist()

            # Step 2: Find similar items for each purchased item (k nearest neighbors)
            recommended_items = []
            for purchased_item in purchased_items:
                # Get k most similar items (exclude self)
                similar_items = item_similarity[purchased_item].drop(purchased_item).sort_values(ascending=False).head(k)
                # Weight items by item similarity (PPT: Weighted Voting)
                recommended_items.extend([(item, similarity_score) for item, similarity_score in similar_items.items()])

            # Step 3: Rank items by total similarity weight, deduplicate, and exclude purchased items
            item_scores = defaultdict(float)
            for item, score in recommended_items:
                if item not in purchased_items:  # Avoid recommending already purchased items
                    item_scores[item] += score

            # Sort items by score (highest first) and select top N
            sorted_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)
            top_recommendations = [item for item, score in sorted_items[:top_n]]

            return top_recommendations, purchased_items
        except Exception as e:
            print(f"Error in item - based collaborative filtering: {e}")
            return [], None
    
    # 4. Model Evaluation (Evaluating Collaborative Filtering)
    # Purpose: Evaluate recommendation performance using train - test split and MSE
  
    def evaluate_recommendation_system(user_item_matrix, similarity_type='user', k=3, top_n=5):
        """
        Evaluate recommendation system by withholding purchases and predicting them (Evaluation Logic)
        """
        try:
            # Split data into train (80%) and test (20%) - simulate withheld purchases
            train_matrix, test_matrix = train_test_split(
                user_item_matrix,
                test_size=0.2,
                random_state=42,
                shuffle=True
            )

            # Calculate similarity on training data
            if similarity_type == 'user':
                similarity = calculate_similarity(train_matrix, similarity_type='user')
            else:
                similarity = calculate_similarity(train_matrix, similarity_type='item')

            if similarity is None:
                return {"MSE": -1, "Precision@{}".format(top_n): -1}

            # Track predictions and actual values
            predictions = []
            actuals = []

            for user_id in test_matrix.index:
                if user_id not in train_matrix.index:
                    continue  # Skip users not in training set

                # Get actual purchased items in test set
                actual_purchases = test_matrix.loc[user_id][test_matrix.loc[user_id] == 1].index.tolist()

                # Generate recommendations
                if similarity_type == 'user':
                    recs, _ = user_based_collaborative_filtering(user_id, train_matrix, similarity, k=k, top_n=top_n)
                else:
                    recs, _ = item_based_collaborative_filtering(user_id, train_matrix, similarity, k=k, top_n=top_n)

                # Convert recommendations to binary (1 = recommended and actually purchased, 0 = otherwise)
                for item in test_matrix.columns:
                    pred = 1 if item in recs else 0
                    actual = 1 if item in actual_purchases else 0
                    predictions.append(pred)
                    actuals.append(actual)

            # Calculate MSE (Evaluation Metrics - Error Calculation)
            mse = mean_squared_error(actuals, predictions) if predictions else -1
            # Calculate precision@top_n (fraction of recommendations that are actual purchases)
            precision = sum([1 for p, a in zip(predictions, actuals) if p == 1 and a == 1]) / sum(predictions) if sum(predictions) > 0 else 0

            return {
                'MSE': round(mse, 4),
                'Precision@{}'.format(top_n): round(precision, 4)
            }
        except Exception as e:
            print(f"Error in model evaluation: {e}")
            return {"MSE": -1, "Precision@{}".format(top_n): -1}


    # Evaluate both user - based and item - based CF
    user_based_metrics = evaluate_recommendation_system(user_item_matrix, similarity_type='user', k=3, top_n=3)
    item_based_metrics = evaluate_recommendation_system(user_item_matrix, similarity_type='item', k=3, top_n=3)

    print("\nEvaluation Metrics (Evaluating Collaborative Filtering):")
    print("User - Based CF:", user_based_metrics)
    print("Item - Based CF:", item_based_metrics)

   
    # 5. Generate Final Recommendations 
    # Purpose: Provide checkout recommendations using both CF methods 
    
    def generate_final_recommendations(
            user_id,
            user_item_matrix,
            user_similarity, 
            item_similarity,
            k=3,
            top_n=5
    ):
        """
        Generate hybrid recommendations (user - based + item - based) aligned with BigBasket's needs:
        - Solve "forgotten items" problem (case)
        - Leverage collaborative filtering 
        """
        try:
            # Get recommendations from both CF methods
            user_based_recs, similar_users = user_based_collaborative_filtering(
                user_id, user_item_matrix, user_similarity, k=k, top_n=top_n
            )
            item_based_recs, purchased_items = item_based_collaborative_filtering(
                user_id, user_item_matrix, item_similarity, k=k, top_n=top_n
            )

            # Combine and rank recommendations (prioritize items recommended by both methods)
            combined_recs = defaultdict(int)
            for idx, rec in enumerate(user_based_recs):
                combined_recs[rec] += (top_n - idx)  # Weight by rank in user - based
            for idx, rec in enumerate(item_based_recs):
                combined_recs[rec] += (top_n - idx)  # Weight by rank in item - based

            # Sort and select top N unique recommendations
            final_recs = sorted(combined_recs.keys(), key=lambda x: combined_recs[x], reverse=True)[:top_n]

            return {
                'user_based_recommendations': user_based_recs,
                'item_based_recommendations': item_based_recs,
                'final_hybrid_recommendations': final_recs,
                'user_purchased_items': purchased_items
            }
        except Exception as e:
            print(f"Error generating final recommendations: {e}")
            return {"user_based_recommendations": [], "item_based_recommendations": [], "final_hybrid_recommendations": [], "user_purchased_items": []}


    # Generate recommendations for sample user M09736 (BigBasket case user)
    sample_user = 'M09736'
    recommendations = generate_final_recommendations(
        sample_user, user_item_matrix, user_similarity, item_similarity, k=3, top_n=5
    )

    print(f"\nRecommendations for User {sample_user} (BigBasket Checkout Scenario):")
    print(f"Purchased Items: {recommendations['user_purchased_items']}")
    print(f"User - Based CF Recommendations: {recommendations['user_based_recommendations']}")
    print(f"Item - Based CF Recommendations: {recommendations['item_based_recommendations']}")
    print(f"Final Hybrid Checkout Recommendations: {recommendations['final_hybrid_recommendations']}")

    
    # 6. Visualization (Support Business Analysis - Similarity & Recommendations)
    
    def visualize_similarity_and_recommendations(item_similarity, top_n=5):
        """Visualize top item - item similarities (key for item - based CF)"""
        try:
            # Get top similar item pairs
            similar_pairs = []
            items = item_similarity.index
            for i in range(len(items)):
                for j in range(i + 1, len(items)):
                    similar_pairs.append((items[i], items[j], item_similarity.iloc[i, j]))
            # Sort and select top N pairs
            similar_pairs.sort(key=lambda x: x[2], reverse=True)
            top_pairs = similar_pairs[:top_n]
            # Plot
            plt.figure(figsize=(10, 6))
            pairs = [f"{p1}\n&{p2}" for p1, p2, _ in top_pairs]
            scores = [score for _, _, score in top_pairs]
            plt.bar(pairs, scores, color='#1f77b4')
            plt.title(f'Top {top_n} Most Similar Item Pairs (Item - Based CF)', fontsize=12)
            plt.xlabel('Item Pairs', fontsize=10)
            plt.ylabel('Cosine Similarity Score', fontsize=10)
            plt.ylim(0, 1)
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Error in visualization: {e}")


    # Execute visualization
    visualize_similarity_and_recommendations(item_similarity, top_n=5) 