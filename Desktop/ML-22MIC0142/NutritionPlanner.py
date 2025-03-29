import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import streamlit as st

# Creating a new dataset with meaningful meal names and increased data size
data = {
    "name": [
        "Grilled Chicken Salad", "Paneer Tikka", "Egg Curry", "Dal Tadka", "Grilled Fish", 
        "Vegetable Stir Fry", "Chickpea Salad", "Tofu Scramble", "Brown Rice & Lentils", "Quinoa Bowl",
        "Oatmeal with Nuts", "Smoothie Bowl", "Pasta with Pesto", "Chicken Soup", "Avocado Toast",
        "Baked Sweet Potato", "Greek Yogurt with Honey", "Protein Shake", "Mushroom Risotto", "Steamed Vegetables"
    ],
    "nutri_energy": np.random.randint(200, 700, 20),
    "nutri_protein": np.random.randint(5, 40, 20),
    "nutri_carbohydrate": np.random.randint(20, 80, 20),
    "nutri_fat": np.random.randint(5, 30, 20)
}
df = pd.DataFrame(data)

# Fill missing numeric values with column mean (not needed here but ensures robustness)
df.fillna(df.mean(numeric_only=True), inplace=True)

# Data Visualization
def visualize_data():
    st.subheader("📊 Data Visualization")
    plt.figure(figsize=(8, 5))
    sns.histplot(df['nutri_energy'], kde=True, bins=30, color="blue")
    st.pyplot(plt)

# K-Means Clustering
X = df[["nutri_energy", "nutri_protein", "nutri_carbohydrate", "nutri_fat"]]
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X)

# Meal Recommendation System
model = NearestNeighbors(n_neighbors=5, metric='euclidean')
model.fit(X)

# Streamlit Web App Interface
def main():
    st.title("🍽️ AI-Powered Nutrition Planner")
    st.subheader("Personalized Meal Recommendations Based on Your Health Goals")

    visualize_data()

    calories = st.slider("Calories Intake", int(df['nutri_energy'].min()), int(df['nutri_energy'].max()), 400)
    protein = st.slider("Protein Intake (g)", int(df['nutri_protein'].min()), int(df['nutri_protein'].max()), 20)
    carbs = st.slider("Carbs Intake (g)", int(df['nutri_carbohydrate'].min()), int(df['nutri_carbohydrate'].max()), 50)
    fat = st.slider("Fat Intake (g)", int(df['nutri_fat'].min()), int(df['nutri_fat'].max()), 15)

    if st.button("Get Meal Recommendation"):
        user_input = [[calories, protein, carbs, fat]]
        distances, indices = model.kneighbors(user_input)
        recommended_meals = df.iloc[indices[0]]

        st.write("✅ **Recommended Meals Based on Your Input:**")
        st.dataframe(recommended_meals[['name', 'nutri_energy', 'nutri_protein', 'nutri_carbohydrate', 'nutri_fat']])

if __name__ == "__main__":
    main()
