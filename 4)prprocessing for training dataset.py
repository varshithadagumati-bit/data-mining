import pandas as pd

# Create a dummy weather.csv file for demonstration
dummy_weather_data = {
    'Date': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04']),
    'Temperature': [10, 12, None, 15],
    'Humidity': [80, 85, 78, 90],
    'Condition': ['Sunny', 'Cloudy', 'Rainy', 'Sunny']
}
df_weather_dummy = pd.DataFrame(dummy_weather_data)
df_weather_dummy.to_csv('weather.csv', index=False)

data = pd.read_csv("weather.csv")

# Handle missing values using the recommended method
data.ffill(inplace=True)

# Convert categorical
data = pd.get_dummies(data, columns=['Condition'])

print(data.head())
