try:
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    import numpy as np

    # Data Load
    df = pd.read_csv('House_Rent_Dataset.csv')

    # Identify and clean problematic data entries
    print('Data Before Cleaning:')
    print(df.head())

    # Remove 'Date' column if present, or any other irrelevant columns
    if 'Date' in df.columns:
        df = df.drop(columns=['Date'])  # Drop the date column

    # Fix entries like '6 out of 8' in numeric columns
    # Assuming it appears in a column such as 'Rating' or 'Floor'
    for col in df.columns:
        if df[col].dtype == 'object':  # Check if the column is of type object (contains strings)
            print(f"Non-numeric values in {col}:")
            print(df[col].unique())  # Show unique values for inspection

            # Attempt to extract the first numeric value if the format is like '6 out of 8'
            df[col] = df[col].str.extract(r'(\d+)').astype(float)

    print("\nData After Cleaning:")
    print(df.head())

    # Handle any NaN values created during the cleaning process
    df.fillna(df.median(numeric_only=True), inplace=True)

    # One-hot encoding for categorical columns (e.g., 'Location' and 'City')
    df = pd.get_dummies(df, columns=['Location', 'City'], drop_first=True)

    print("\nData After Preprocessing:")
    print(df.head())

    # Check for NaN and Infinite values again after cleaning
    print(f"NaN values in X: {df.isna().sum().sum()}")
    print(f"Infinite values in numeric columns: {np.isinf(df.select_dtypes(include=[np.number])).sum().sum()}")

    # Features and Target Variable
    X = df.drop('Price', axis=1)
    y = df['Price']

    # Split Data into Training and Testing Sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Feature Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Model Training
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predictions for Test Data
    y_pred = model.predict(X_test)

    # Model Evaluation
    print(f"MAE: {mean_absolute_error(y_test, y_pred)}")
    print(f"MSE: {mean_squared_error(y_test, y_pred)}")
    print(f"R2 Score: {r2_score(y_test, y_pred)}")

    # Taking User Input
    print("\n--- Enter details of the house to predict price ---")
    house_size = float(input("Enter house size (in sqft): "))
    bedrooms = int(input("Enter number of bedrooms: "))
    location = input("Enter location (match it with dataset locations): ")

    # Prepare user input data
    user_data = {
        'Size': [house_size],
        'Bedroom': [bedrooms]
    }

    # Handling the location as a one-hot encoded value
    for loc in df.columns[df.columns.str.startswith('Location_')]:
        if f'Location_{location}' == loc:
            user_data[loc] = [1]
        else:
            user_data[loc] = [0]

    # Convert user data to a DataFrame
    user_df = pd.DataFrame(user_data)

    # Feature scaling for user input
    user_scaled = scaler.transform(user_df)

    # Predict the house price based on user input
    predicted_price = model.predict(user_scaled)

    print(f"Estimated House Price: ${predicted_price[0]:.2f}")

except ImportError as e:
    print(f"Debugging failed: {e}")
except Exception as e:
    print(f"Error: {e}")
