try:
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    import numpy as np

    df = pd.read_csv('House_Rent_Dataset.csv')

    print('Data Before Cleaning:')
    print(df.head())

    df = df[['Size', 'Bedroom', 'City', 'Location', 'Price']]
    df.dropna(subset=['Size', 'Bedroom', 'City', 'Location', 'Price'], inplace=True)
    df = pd.get_dummies(df, columns=['City', 'Location'], drop_first=True)

    print("\nData After Cleaning:")
    print(df.head())

    X = df.drop('Price', axis=1) 
    y = df['Price'] 

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_predict = model.predict(X_test)

    print(f"\nMean Absolute Error: {mean_absolute_error(y_test, y_predict)}")
    print(f"Mean Squared Error: {mean_squared_error(y_test, y_predict)}")
    print(f"R2 Score: {r2_score(y_test, y_predict)}")

    house_size = float(input("\nEnter House Size (in sqft): "))
    bedrooms = int(input("Enter No of Bedrooms: "))
    city = input("Enter House City: ")
    location = input("Enter House Location: ")

    user_data = {
        'Size': [house_size],
        'Bedroom': [bedrooms]
    }

    for col in df.columns[df.columns.str.startswith('City_')]:
        user_data[col] = [1 if f'City_{city}' == col else 0]

    for col in df.columns[df.columns.str.startswith('Location_')]:
        user_data[col] = [1 if f'Location_{location}' == col else 0]

    user_df = pd.DataFrame(user_data)
    user_scaled = scaler.transform(user_df)
    predicted_price = model.predict(user_scaled)

    print(f"\nHouse Price: ${predicted_price[0]:.2f}")

except ImportError as e:
    print(f"Debugging failed: {e}")

except Exception as e:
    print(f"Error: {e}")
