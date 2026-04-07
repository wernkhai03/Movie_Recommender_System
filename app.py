# --- UPDATE THE ACCURACY FUNCTION ---
def get_user_rmse(user_ratings, all_ratings):
    if user_ratings.empty:
        return 0.0
    # Compare this user's actual ratings to the global average
    # This shows how "unique" or "difficult to predict" this specific user is
    actual = user_ratings['rating']
    predicted = [all_ratings['rating'].mean()] * len(actual)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    return round(rmse, 4)

# --- UPDATE THE USER ID TAB ---
with tab2:
    u_id = st.number_input("Enter User ID:", min_value=1, step=1)
    if st.button("Get My Recommendations"):
        user_history = ratings[ratings['userId'] == u_id]
        
        if not user_history.empty:
            # DYNAMIC RMSE: Calculate accuracy specifically for THIS user
            current_rmse = get_user_rmse(user_history, ratings)
            st.sidebar.metric("User Prediction Error (RMSE)", current_rmse)
            st.sidebar.caption(f"This score shows how much User {u_id}'s tastes differ from the average.")
            
            # ... (rest of your recommendation logic) ...
