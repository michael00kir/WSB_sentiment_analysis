import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import seaborn as sns
import json
from datetime import datetime
import re
from collections import Counter
import matplotlib.gridspec as gridspec

# Set styling for all plots
plt.style.use('fivethirtyeight')
sns.set_palette("viridis")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 12

# Function to safely parse JSON strings with either single or double quotes
def safe_json_parse(s):
    if not s or s == '[]':
        return []
    # Replace single quotes with double quotes for JSON parsing
    fixed_str = s.replace("'", '"')
    try:
        return json.loads(fixed_str)
    except:
        return []

# Function to create date string for x-axis
def format_date_tick(x, pos=None):
    dt = mdates.num2date(x)
    return dt.strftime('%m-%d')

# Load the data
def load_and_prepare_data(file_path):
    df = pd.read_csv(file_path)
    
    # Convert created_utc to datetime
    df['created_utc'] = pd.to_datetime(df['created_utc'])
    df['date'] = df['created_utc'].dt.date
    
    # Parse JSON columns
    df['tickers_list'] = df['tickers'].apply(safe_json_parse)
    df['wsb_terms_list'] = df['wsb_terms'].apply(safe_json_parse)
    df['emojis_list'] = df['emojis'].apply(safe_json_parse)
    
    return df

# 1. Overall Sentiment Distribution
def plot_sentiment_distribution(df):
    plt.figure(figsize=(10, 6))
    sentiment_counts = df['sentiment'].value_counts()
    colors = {'positive': '#2ecc71', 'neutral': '#3498db', 'negative': '#e74c3c'}
    
    ax = sentiment_counts.plot(kind='bar', color=[colors[sent] for sent in sentiment_counts.index])
    
    total = len(df)
    for i, count in enumerate(sentiment_counts):
        percentage = count / total * 100
        ax.text(i, count + 5, f"{percentage:.1f}%", ha='center')
    
    plt.title('Distribution of Sentiment', fontsize=16)
    plt.ylabel('Count', fontsize=14)
    plt.xlabel('Sentiment', fontsize=14)
    plt.xticks(rotation=0)
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('sentiment_distribution.png', dpi=300)
    plt.close()

# 2. Sentiment Over Time
def plot_sentiment_over_time(df):
    # Group by date and calculate sentiment metrics
    daily_sentiment = df.groupby('date').agg({
        'sentiment': lambda x: x.value_counts().to_dict(),
        'sentiment_score': 'mean',
        'id': 'count'
    }).reset_index()
    
    # Extract positive, neutral, negative counts
    for sent in ['positive', 'neutral', 'negative']:
        daily_sentiment[sent] = daily_sentiment['sentiment'].apply(
            lambda x: x.get(sent, 0) if isinstance(x, dict) else 0
        )
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), gridspec_kw={'height_ratios': [2, 1]})
    
    # Plot stacked bar chart for sentiment counts
    dates = mdates.date2num(daily_sentiment['date'])
    width = 0.8
    
    # Create stacked bars
    ax1.bar(dates, daily_sentiment['negative'], width, label='Negative', color='#e74c3c', alpha=0.8)
    ax1.bar(dates, daily_sentiment['neutral'], width, bottom=daily_sentiment['negative'], 
            label='Neutral', color='#3498db', alpha=0.8)
    ax1.bar(dates, daily_sentiment['positive'], width, 
            bottom=daily_sentiment['negative'] + daily_sentiment['neutral'], 
            label='Positive', color='#2ecc71', alpha=0.8)
    
    # Add post count as a line
    ax1b = ax1.twinx()
    ax1b.plot(dates, daily_sentiment['id'], color='#9b59b6', linewidth=2, marker='o', markersize=4, label='Post Count')
    ax1b.set_ylabel('Number of Posts', color='#9b59b6', fontsize=14)
    ax1b.tick_params(axis='y', labelcolor='#9b59b6')
    
    # Line plot for average sentiment score
    ax2.plot(dates, daily_sentiment['sentiment_score'], color='#e67e22', linewidth=2, marker='o', markersize=4)
    ax2.set_ylabel('Avg Sentiment Score', color='#e67e22', fontsize=14)
    ax2.tick_params(axis='y', labelcolor='#e67e22')
    ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7)
    ax2.set_ylim([0.4, 0.8])
    
    # Format x-axis for both subplots
    for ax in [ax1, ax2]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        ax.grid(axis='y', alpha=0.3)
    
    ax1.set_title('Sentiment Distribution and Post Activity Over Time', fontsize=16)
    ax1.set_ylabel('Count of Posts by Sentiment', fontsize=14)
    ax1.legend(loc='upper left')
    ax1b.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig('sentiment_over_time.png', dpi=300)
    plt.close()

# 3. Weekly Sentiment Trends
def plot_weekly_sentiment_trends(df):
    # Add week info
    df['year_week'] = df['created_utc'].dt.strftime('%Y-W%U')
    
    # Group by week
    weekly = df.groupby('year_week').agg({
        'id': 'count',
        'sentiment_score': 'mean',
        'sentiment': lambda x: x.value_counts().to_dict(),
        'score': 'sum',
        'num_comments': 'sum'
    }).reset_index()
    
    # Extract sentiment counts
    for sent in ['positive', 'neutral', 'negative']:
        weekly[sent] = weekly['sentiment'].apply(
            lambda x: x.get(sent, 0) if isinstance(x, dict) else 0
        )
    
    # Calculate sentiment proportions
    for sent in ['positive', 'neutral', 'negative']:
        weekly[f'{sent}_prop'] = weekly[sent] / weekly['id']
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [1, 2]})
    
    # Line plot for weekly post count and sentiment score
    ax1.plot(range(len(weekly)), weekly['id'], 'o-', color='#9b59b6', linewidth=3, label='Post Count')
    ax1.set_ylabel('Post Count', color='#9b59b6', fontsize=14)
    ax1.tick_params(axis='y', labelcolor='#9b59b6')
    
    ax1b = ax1.twinx()
    ax1b.plot(range(len(weekly)), weekly['sentiment_score'], 'o-', color='#e67e22', linewidth=3, label='Avg Sentiment')
    ax1b.set_ylabel('Sentiment Score', color='#e67e22', fontsize=14)
    ax1b.tick_params(axis='y', labelcolor='#e67e22')
    
    # Area chart for sentiment proportions
    ax2.stackplot(range(len(weekly)), 
                  weekly['negative_prop'], 
                  weekly['neutral_prop'], 
                  weekly['positive_prop'],
                  labels=['Negative', 'Neutral', 'Positive'],
                  colors=['#e74c3c', '#3498db', '#2ecc71'],
                  alpha=0.7)
    
    # Set x-ticks
    for ax in [ax1, ax2]:
        ax.set_xticks(range(len(weekly)))
        ax.set_xticklabels(weekly['year_week'], rotation=45)
        ax.grid(axis='y', alpha=0.3)
    
    ax1.set_title('Weekly WSB Sentiment Trends', fontsize=16)
    ax1.legend(loc='upper left')
    ax1b.legend(loc='upper right')
    
    ax2.set_ylabel('Proportion of Sentiment', fontsize=14)
    ax2.set_xlabel('Week', fontsize=14)
    ax2.legend(loc='upper left')
    
    plt.tight_layout()
    plt.savefig('weekly_sentiment_trends.png', dpi=300)
    plt.close()

# 4. Top Mentioned Tickers with Sentiment
def plot_top_tickers(df):
    # Explode tickers list to get one row per ticker
    ticker_df = df.explode('tickers_list')
    ticker_df = ticker_df[ticker_df['tickers_list'].notna()]
    
    if len(ticker_df) == 0:
        return  # No tickers found
        
    # Get top tickers
    top_tickers = ticker_df['tickers_list'].value_counts().head(10)
    
    # Calculate sentiment for each top ticker
    ticker_sentiment = {}
    
    for ticker in top_tickers.index:
        subset = ticker_df[ticker_df['tickers_list'] == ticker]
        sent_counts = subset['sentiment'].value_counts()
        ticker_sentiment[ticker] = {
            'count': len(subset),
            'avg_score': subset['sentiment_score'].mean(),
            'positive': sent_counts.get('positive', 0),
            'neutral': sent_counts.get('neutral', 0),
            'negative': sent_counts.get('negative', 0)
        }
    
    # Create DataFrame from ticker sentiment dict
    ticker_sent_df = pd.DataFrame.from_dict(ticker_sentiment, orient='index')
    ticker_sent_df = ticker_sent_df.sort_values('count', ascending=False)
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # Bar chart for ticker counts
    ticker_sent_df['count'].plot(kind='bar', ax=ax1, color='#3498db')
    ax1.set_title('Top 10 Mentioned Tickers', fontsize=16)
    ax1.set_ylabel('Mention Count', fontsize=14)
    ax1.set_xlabel('Ticker', fontsize=14)
    
    # Stacked bar chart for sentiment
    sentiment_data = ticker_sent_df[['positive', 'neutral', 'negative']]
    sentiment_data.plot(kind='bar', stacked=True, ax=ax2, 
                       color=['#2ecc71', '#3498db', '#e74c3c'])
    
    ax2.set_title('Sentiment Distribution by Ticker', fontsize=16)
    ax2.set_ylabel('Count', fontsize=14)
    ax2.set_xlabel('Ticker', fontsize=14)
    
    for ax in [ax1, ax2]:
        ax.grid(axis='y', alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    plt.savefig('top_tickers_sentiment.png', dpi=300)
    plt.close()

# 5. WSB Terms Analysis
def plot_wsb_terms(df):
    # Explode wsb_terms list to get one row per term
    terms_df = df.explode('wsb_terms_list')
    terms_df = terms_df[terms_df['wsb_terms_list'].notna()]
    
    if len(terms_df) == 0:
        return  # No terms found
        
    # Get top terms
    top_terms = terms_df['wsb_terms_list'].value_counts().head(15)
    
    # Calculate sentiment for each top term
    term_sentiment = {}
    
    for term in top_terms.index:
        subset = terms_df[terms_df['wsb_terms_list'] == term]
        sent_counts = subset['sentiment'].value_counts()
        term_sentiment[term] = {
            'count': len(subset),
            'avg_score': subset['sentiment_score'].mean(),
            'positive': sent_counts.get('positive', 0) / len(subset),
            'neutral': sent_counts.get('neutral', 0) / len(subset),
            'negative': sent_counts.get('negative', 0) / len(subset)
        }
    
    # Create DataFrame from term sentiment dict
    term_sent_df = pd.DataFrame.from_dict(term_sentiment, orient='index')
    term_sent_df = term_sent_df.sort_values('count', ascending=False)
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # Bar chart for term counts
    term_sent_df['count'].plot(kind='bar', ax=ax1, color='#2980b9')
    ax1.set_title('Top 15 WSB Terms', fontsize=16)
    ax1.set_ylabel('Mention Count', fontsize=14)
    ax1.set_xlabel('Term', fontsize=14)
    
    # Horizontal stacked bar for sentiment proportions
    sentiment_props = term_sent_df[['positive', 'neutral', 'negative']]
    sentiment_props.plot(kind='barh', stacked=True, ax=ax2, 
                         color=['#2ecc71', '#3498db', '#e74c3c'])
    
    ax2.set_title('Sentiment Proportion by WSB Term', fontsize=16)
    ax2.set_xlabel('Proportion', fontsize=14)
    ax2.set_ylabel('Term', fontsize=14)
    
    for ax in [ax1, ax2]:
        ax.grid(axis='y', alpha=0.3)
    
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig('wsb_terms_sentiment.png', dpi=300)
    plt.close()

# 6. Emoji Analysis
def plot_emoji_analysis(df):
    # Explode emojis list to get one row per emoji
    emoji_df = df.explode('emojis_list')
    emoji_df = emoji_df[emoji_df['emojis_list'].notna()]
    
    if len(emoji_df) == 0:
        print("No emoji data found.")
        return  # No emojis found
    
    # Get top emojis
    top_emojis = emoji_df['emojis_list'].value_counts().head(10)
    
    # Calculate sentiment for each top emoji
    emoji_sentiment = {}
    
    for emoji in top_emojis.index:
        subset = emoji_df[emoji_df['emojis_list'] == emoji]
        sent_counts = subset['sentiment'].value_counts()
        emoji_sentiment[emoji] = {
            'count': len(subset),
            'avg_score': subset['sentiment_score'].mean(),
            'positive': sent_counts.get('positive', 0),
            'neutral': sent_counts.get('neutral', 0),
            'negative': sent_counts.get('negative', 0)
        }
    
    # Create DataFrame from emoji sentiment dict
    emoji_sent_df = pd.DataFrame.from_dict(emoji_sentiment, orient='index')
    emoji_sent_df = emoji_sent_df.sort_values('count', ascending=False)
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # Bar chart for emoji counts
    ax1.bar(range(len(emoji_sent_df)), emoji_sent_df['count'], color=sns.color_palette("viridis", len(emoji_sent_df)))
    ax1.set_xticks(range(len(emoji_sent_df)))
    ax1.set_xticklabels(emoji_sent_df.index, fontsize=16)
    ax1.set_title('Top 10 Emojis in WSB Posts', fontsize=16)
    ax1.set_ylabel('Count', fontsize=14)
    
    # Pie chart for sentiment distribution with emojis
    emoji_sent = emoji_sent_df[['positive', 'neutral', 'negative']].sum()
    ax2.pie(emoji_sent, labels=['Positive', 'Neutral', 'Negative'],
            autopct='%1.1f%%', startangle=90, colors=['#2ecc71', '#3498db', '#e74c3c'],
            wedgeprops={'edgecolor': 'w', 'linewidth': 1})
    ax2.set_title('Overall Sentiment in Posts with Emojis', fontsize=16)
    
    plt.tight_layout()
    plt.savefig('emoji_analysis.png', dpi=300)
    plt.close()

# 7. Time of Day Analysis
def plot_time_of_day_analysis(df):
    # Extract hour from datetime
    df['hour'] = df['created_utc'].dt.hour
    
    # Group by hour
    hourly = df.groupby('hour').agg({
        'id': 'count',
        'sentiment_score': 'mean',
        'score': 'mean',
        'num_comments': 'mean'
    }).reset_index()
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [1.5, 1]})
    
    # Line plot for post count by hour
    ax1.plot(hourly['hour'], hourly['id'], 'o-', color='#9b59b6', linewidth=3, markersize=8, label='Post Count')
    ax1.set_ylabel('Number of Posts', color='#9b59b6', fontsize=14)
    ax1.tick_params(axis='y', labelcolor='#9b59b6')
    ax1.set_title('WSB Activity and Engagement by Hour of Day', fontsize=16)
    
    # Twin axis for sentiment score
    ax1b = ax1.twinx()
    ax1b.plot(hourly['hour'], hourly['sentiment_score'], 'o-', color='#e67e22', linewidth=3, markersize=8, label='Avg Sentiment')
    ax1b.set_ylabel('Sentiment Score', color='#e67e22', fontsize=14)
    ax1b.tick_params(axis='y', labelcolor='#e67e22')
    
    # Line plots for engagement metrics
    ax2.plot(hourly['hour'], hourly['score'], 'o-', color='#2980b9', linewidth=2, label='Avg Score')
    ax2.plot(hourly['hour'], hourly['num_comments'], 'o-', color='#27ae60', linewidth=2, label='Avg Comments')
    ax2.set_ylabel('Average Engagement', fontsize=14)
    ax2.set_xlabel('Hour of Day (UTC)', fontsize=14)
    
    # Set x-ticks for both plots
    for ax in [ax1, ax2]:
        ax.set_xticks(range(0, 24, 2))
        ax.set_xlim(-0.5, 23.5)
        ax.grid(alpha=0.3)
    
    # Add legends
    ax1.legend(loc='upper left')
    ax1b.legend(loc='upper right')
    ax2.legend(loc='best')
    
    plt.tight_layout()
    plt.savefig('time_of_day_analysis.png', dpi=300)
    plt.close()

# 8. Correlation Matrix of Key Metrics
def plot_correlation_matrix(df):
    # Select numeric columns
    numeric_cols = ['sentiment_score', 'score', 'num_comments', 'title_length']
    
    # Add title length
    df['title_length'] = df['title'].str.len()
    
    # Calculate correlation
    corr_matrix = df[numeric_cols].corr()
    
    # Plot correlation matrix
    plt.figure(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='viridis',
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    
    plt.title('Correlation Between Key WSB Post Metrics', fontsize=16)
    plt.tight_layout()
    plt.savefig('correlation_matrix.png', dpi=300)
    plt.close()

# 9. Title Length vs. Engagement
def plot_title_vs_engagement(df):
    # Calculate title length
    df['title_length'] = df['title'].str.len()
    
    # Create bins for title length
    bins = [0, 50, 100, 150, 200, 250, 300, 350, 400]
    df['title_length_bin'] = pd.cut(df['title_length'], bins=bins)
    
    # Group by bins
    title_stats = df.groupby('title_length_bin').agg({
        'id': 'count',
        'score': 'mean',
        'num_comments': 'mean',
        'sentiment_score': 'mean'
    }).reset_index()
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot for post count by title length
    ax1.bar(range(len(title_stats)), title_stats['id'], color='#3498db')
    ax1.set_ylabel('Number of Posts', fontsize=14)
    ax1.set_title('Post Count and Engagement by Title Length', fontsize=16)
    
    # Plot for engagement metrics
    ax2.plot(range(len(title_stats)), title_stats['score'], 'o-', color='#e74c3c', linewidth=2, label='Avg Score')
    ax2.plot(range(len(title_stats)), title_stats['num_comments'], 'o-', color='#2ecc71', linewidth=2, label='Avg Comments')
    ax2.set_ylabel('Average Engagement', fontsize=14)
    ax2.legend()
    
    # Set x-ticks for both plots
    bin_labels = [f'{b.left}-{b.right}' for b in title_stats['title_length_bin']]
    for ax in [ax1, ax2]:
        ax.set_xticks(range(len(title_stats)))
        ax.set_xticklabels(bin_labels, rotation=45)
        ax.grid(axis='y', alpha=0.3)
    
    ax2.set_xlabel('Title Length (characters)', fontsize=14)
    
    plt.tight_layout()
    plt.savefig('title_vs_engagement.png', dpi=300)
    plt.close()

# 10. Community Award Analysis (if available)
def plot_award_analysis(df):
    # Check if award data is available
    if 'total_awards_received' not in df.columns:
        print("No award data available.")
        return
    
    # Filter posts with awards
    awarded_posts = df[df['total_awards_received'] > 0]
    
    if len(awarded_posts) == 0:
        print("No posts with awards found.")
        return
        
    # Group by sentiment
    award_by_sentiment = awarded_posts.groupby('sentiment').agg({
        'id': 'count',
        'total_awards_received': ['sum', 'mean'],
        'score': 'mean'
    })
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Bar chart for award count by sentiment
    sentiment_colors = {'positive': '#2ecc71', 'neutral': '#3498db', 'negative': '#e74c3c'}
    award_by_sentiment[('total_awards_received', 'sum')].plot(
        kind='bar', ax=ax1, color=[sentiment_colors[s] for s in award_by_sentiment.index]
    )
    ax1.set_title('Total Awards by Sentiment', fontsize=16)
    ax1.set_ylabel('Number of Awards', fontsize=14)
    ax1.set_xlabel('Sentiment', fontsize=14)
    
    # Bar chart for average awards per post by sentiment
    award_by_sentiment[('total_awards_received', 'mean')].plot(
        kind='bar', ax=ax2, color=[sentiment_colors[s] for s in award_by_sentiment.index]
    )
    ax2.set_title('Average Awards per Post by Sentiment', fontsize=16)
    ax2.set_ylabel('Avg Awards per Post', fontsize=14)
    ax2.set_xlabel('Sentiment', fontsize=14)
    
    for ax in [ax1, ax2]:
        ax.grid(axis='y', alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=0)
    
    plt.tight_layout()
    plt.savefig('award_analysis.png', dpi=300)
    plt.close()

# Main function to run all analyses
def main(file_path):
    print(f"Loading data from {file_path}...")
    df = load_and_prepare_data(file_path)
    print(f"Loaded {len(df)} posts.")
    
    print("Generating visualizations...")
    
    print("1. Plotting sentiment distribution...")
    plot_sentiment_distribution(df)
    
    print("2. Plotting sentiment over time...")
    plot_sentiment_over_time(df)
    
    print("3. Plotting weekly sentiment trends...")
    plot_weekly_sentiment_trends(df)
    
    print("4. Plotting top tickers with sentiment...")
    plot_top_tickers(df)
    
    print("5. Plotting WSB terms analysis...")
    plot_wsb_terms(df)
    
    print("6. Plotting emoji analysis...")
    plot_emoji_analysis(df)
    
    print("7. Plotting time of day analysis...")
    plot_time_of_day_analysis(df)
    
    print("8. Plotting correlation matrix...")
    plot_correlation_matrix(df)
    
    print("9. Plotting title length vs engagement...")
    plot_title_vs_engagement(df)
    
    print("10. Plotting award analysis (if available)...")
    plot_award_analysis(df)
    
    print("All visualizations completed!")

# Run the analysis if this script is executed directly
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = input("Please enter the path to your WSB data CSV file: ")
    
    main(file_path)
