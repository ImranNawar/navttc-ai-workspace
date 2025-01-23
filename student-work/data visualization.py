import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Load the dataset
file_path = 'C:/Users/choice/Desktop/ChatGPT_Reviews.csv'
df = pd.read_csv(file_path)



# Handle missing values in the 'Review' column
df['Review'].fillna('No Review', inplace=True)

# Convert 'Review Date' to datetime format
df['Review Date'] = pd.to_datetime(df['Review Date'], errors='coerce')


# 2. Exploratory Data Analysis


print(df.info())
print(df.describe(include='all'))


# 3. Data Visualization

# 3.1 Ratings Distribution
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='Ratings', palette='viridis')
plt.title('Ratings Distribution')
plt.xlabel('Ratings')
plt.ylabel('Count')
plt.show()






df['Review Day'] = df['Review Date'].dt.date

daily_reviews = df.groupby('Review Day').size()

plt.figure(figsize=(12, 6))
plt.plot(daily_reviews.index, daily_reviews.values, color='blue')
plt.title('Number of Reviews Over Time')
plt.xlabel('Date')
plt.ylabel('Number of Reviews')
plt.grid()
plt.show()

# 3.4 Pie Chart of Ratings Proportions
ratings_proportions = df['Ratings'].value_counts()

plt.figure(figsize=(8, 8))
ratings_proportions.plot.pie(autopct='%1.1f%%', startangle=90, colors=sns.color_palette('pastel'))
plt.title('Ratings Proportions')
plt.ylabel('')
plt.show()

# Save the cleaned dataset (optional)
cleaned_file_path = '/mnt/data/Cleaned_ChatGPT_Reviews.csv'
df.to_csv(cleaned_file_path, index=False)
print(f"Cleaned dataset saved to: {cleaned_file_path}")
