import matplotlib.pyplot as plt
import seaborn as sns

def plot_rating_distribution(df):
    """Plot distribution of ratings."""
    sns.histplot(df['stars'], bins=5)
    plt.title('Rating Distribution')
    plt.xlabel('Stars')
    plt.ylabel('Frequency')
    plt.show()

def plot_trend_over_time(df):
    """Plot trends in ratings over time."""
    df['year'] = df['date'].dt.year
    trend = df.groupby('year')['stars'].mean()
    trend.plot(marker='o')
    plt.title('Average Ratings Over Time')
    plt.xlabel('Year')
    plt.ylabel('Average Stars')
    plt.show()
