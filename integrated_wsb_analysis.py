import praw
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from tqdm import tqdm
import datetime
import re
import nltk
from nltk.corpus import stopwords
from collections import Counter
import yfinance as yf
import emoji
import imojify
import pip
from collections import defaultdict
from matplotlib.font_manager import FontProperties

prop = FontProperties(fname='/System/Library/Fonts/Apple Color Emoji.ttc')

# Import our custom WSB preprocessor
from wsb_preprocessing import WSBPreprocessor

# Download required NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

SECRET="zGaknavYBMgF10Y19C_vZRSS-iuelg"
APP_ID="FgIgXFKcGIuADtmeKBnCBw"


# Set up Reddit API credentials
reddit = praw.Reddit(
    client_id=APP_ID,
    client_secret=SECRET,
    user_agent="Comment Extraction",
)

# Function to get posts from WallStreetBets
def get_wsb_posts(limit=10000, time_filter="year", comment_limit=10000):
    print(f"🚀 Starting to fetch up to {limit} posts from r/wallstreetbets ({time_filter})...")
    subreddit = reddit.subreddit("wallstreetbets")
    posts = []
    post_count = 0

    # Get a generator for the posts
    post_generator = subreddit.top(time_filter=time_filter, limit=limit)

    # Process posts with a progress counter
    for post in post_generator:
        post_count += 1
        print(f"[{post_count}/{limit}] Processing post: {post.title[:50]}{'...' if len(post.title) > 50 else ''}")

        # Extract post data
        post_data = {
            "title": post.title,
            "score": post.score,
            "id": post.id,
            "url": post.url,
            "num_comments": post.num_comments,
            "created_utc": datetime.datetime.fromtimestamp(post.created_utc),
            "selftext": post.selftext,
            "comments": []
        }

        # Show comment count
        print(f"  💬 Found {post.num_comments} comments. Loading comment tree...")

        # Ensure all comments are loaded by replacing the CommentForest with the list of comments
        post.comments.replace_more(limit=0)

        # Get top-level comments (first layer)
        top_comments = list(post.comments)[:comment_limit]
        print(f"  📊 Processing {len(top_comments)} top-level comments...")

        for i, top_comment in enumerate(top_comments):
            if i % 5 == 0:  # Show progress every 5 comments
                print(f"  ⏳ Processing top-level comment {i+1}/{len(top_comments)}")

            top_comment_data = {
                "id": top_comment.id,
                "author": str(top_comment.author) if top_comment.author else "[deleted]",
                "body": top_comment.body,
                "score": top_comment.score,
                "created_utc": datetime.datetime.fromtimestamp(top_comment.created_utc),
                "replies": []
            }

            # Get second layer comments (replies to top comments)
            # Ensure all replies are loaded
            top_comment.replies.replace_more(limit=0)

            for second_comment in list(top_comment.replies)[:comment_limit]:
                second_comment_data = {
                    "id": second_comment.id,
                    "author": str(second_comment.author) if second_comment.author else "[deleted]",
                    "body": second_comment.body,
                    "score": second_comment.score,
                    "created_utc": datetime.datetime.fromtimestamp(second_comment.created_utc),
                    "replies": []
                }

                # Get third layer comments (replies to second layer)
                # Ensure all replies are loaded
                second_comment.replies.replace_more(limit=0)

                for third_comment in list(second_comment.replies)[:comment_limit]:
                    third_comment_data = {
                        "id": third_comment.id,
                        "author": str(third_comment.author) if third_comment.author else "[deleted]",
                        "body": third_comment.body,
                        "score": third_comment.score,
                        "created_utc": datetime.datetime.fromtimestamp(third_comment.created_utc)
                    }

                    second_comment_data["replies"].append(third_comment_data)

                top_comment_data["replies"].append(second_comment_data)

            post_data["comments"].append(top_comment_data)

        posts.append(post_data)
        print(f"✅ Completed post {post_count} with {len(post_data['comments'])} comments and their replies")

    print(f"🏁 Done! Successfully processed {post_count} posts from r/wallstreetbets")
    print(f"💾 Creating DataFrame...")

    df = pd.DataFrame(posts)
    print(f"📋 Final DataFrame shape: {df.shape}")

    return df

# Load FinBERT model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")

# Function to get sentiment from FinBERT
def get_finbert_sentiment(text):
    """Get financial sentiment using FinBERT"""
    if pd.isna(text) or text == "":
        return {"label": "neutral", "score": 1.0}

    # Use the tokenizer to handle truncation properly
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )

    try:
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

        # FinBERT labels: 0 = negative, 1 = neutral, 2 = positive
        sentiment_score = predictions[0].tolist()
        max_score_index = sentiment_score.index(max(sentiment_score))

        labels = ["negative", "neutral", "positive"]
        sentiment_label = labels[max_score_index]

        return {"label": sentiment_label, "score": max(sentiment_score)}
    except RuntimeError as e:
        print(f"Error processing text: {e}")
        # Return neutral sentiment if there's an error
        return {"label": "neutral", "score": 1.0}

# Function to blend FinBERT and WSB-specific sentiment
def blend_sentiments(finbert_sentiment, wsb_sentiment_score):
    """Combine FinBERT sentiment with WSB-specific sentiment for a more accurate result"""
    # Convert FinBERT sentiment to numerical score (-1 to 1)
    finbert_numerical = {
        "positive": 1.0,
        "neutral": 0.0,
        "negative": -1.0
    }[finbert_sentiment["label"]] * finbert_sentiment["score"]

    # Normalize WSB sentiment to similar scale
    wsb_numerical = max(min(wsb_sentiment_score / 3.0, 1.0), -1.0)

    # Weight the two scores (can be adjusted)
    finbert_weight = 0.6
    wsb_weight = 0.4

    combined_score = (finbert_numerical * finbert_weight) + (wsb_numerical * wsb_weight)

    # Convert back to categorical
    if combined_score > 0.2:
        return {"label": "positive", "score": abs(combined_score)}
    elif combined_score < -0.2:
        return {"label": "negative", "score": abs(combined_score)}
    else:
        return {"label": "neutral", "score": 1.0 - abs(combined_score)}

# Main function
def analyze_wsb_sentiment(limit=10000, time_filter="year"):
    """Analyzes sentiment of WallStreetBets posts with WSB-specific language understanding"""
    print(f"Fetching {limit} posts from r/wallstreetbets ({time_filter})...")
    df = get_wsb_posts(limit=limit, time_filter=time_filter)
    df.to_csv('rawData.csv', index=False)
    print(f"Found {len(df)} posts. Analyzing sentiment...")

    # Combine title and selftext for better context
    df['full_text'] = df['title'] + " " + df['selftext'].fillna("")

    # Initialize our WSB preprocessor
    wsb_processor = WSBPreprocessor()

    # Process the posts with WSB-specific language understanding
    print("Applying WSB language preprocessing...")
    df = wsb_processor.process_dataframe(df)
    df.to_csv('ProcessedData.csv', index=False)

    # Analyze sentiment with FinBERT
    print("Applying FinBERT sentiment analysis...")
    finbert_results = []
    for text in tqdm(df['processed_text']):
        finbert_results.append(get_finbert_sentiment(text))

    # Add FinBERT sentiment results to dataframe
    df['finbert_sentiment'] = [r['label'] for r in finbert_results]
    df['finbert_score'] = [r['score'] for r in finbert_results]

    # Blend the sentiments
    print("Blending FinBERT and WSB-specific sentiment...")
    blended_results = []
    for i, row in df.iterrows():
        blended = blend_sentiments(
            {"label": row['finbert_sentiment'], "score": row['finbert_score']},
            row['wsb_sentiment_score']
        )
        blended_results.append(blended)

    # Add blended sentiment to dataframe
    df['sentiment'] = [r['label'] for r in blended_results]
    df['sentiment_score'] = [r['score'] for r in blended_results]

    # Calculate sentiment statistics and comparison
    finbert_counts = df['finbert_sentiment'].value_counts()
    wsb_counts = df['wsb_sentiment'].value_counts()
    blended_counts = df['sentiment'].value_counts()

    print("\nSentiment Distribution Comparison:")
    print("FinBERT vs WSB-specific vs Blended:")
    for label in ["positive", "neutral", "negative"]:
        finbert_pct = finbert_counts.get(label, 0) / len(df) * 100
        wsb_pct = wsb_counts.get(label, 0) / len(df) * 100
        blended_pct = blended_counts.get(label, 0) / len(df) * 100
        print(f"{label}: FinBERT={finbert_pct:.1f}%, WSB={wsb_pct:.1f}%, Blended={blended_pct:.1f}%")

    # Generate WSB language report
    wsb_report = wsb_processor.generate_summary_report(df)

    print("\nWSB Language Analysis:")
    print(f"Top WSB Terms: {wsb_report['top_wsb_terms'][:20]}")
    print(f"Top Emojis: {wsb_report['top_emojis'][:20]}")
    print(f"Top Tickers: {wsb_report['top_tickers'][:20]}")

    # Visualize sentiment comparison
    plt.figure(figsize=(15, 6))

    # Create a DataFrame for plotting
    plot_data = pd.DataFrame({
        'FinBERT': [finbert_counts.get('positive', 0), finbert_counts.get('neutral', 0), finbert_counts.get('negative', 0)],
        'WSB': [wsb_counts.get('positive', 0), wsb_counts.get('neutral', 0), wsb_counts.get('negative', 0)],
        'Blended': [blended_counts.get('positive', 0), blended_counts.get('neutral', 0), blended_counts.get('negative', 0)]
    }, index=['Positive', 'Neutral', 'Negative'])

    plot_data.plot(kind='bar', figsize=(12, 6))
    plt.title(f"Sentiment Analysis Comparison for WallStreetBets Posts ({time_filter})")
    plt.xlabel("Sentiment")
    plt.ylabel("Number of Posts")
    plt.legend(title="Method")
    plt.xticks(rotation=0)

    # Add count labels
    for container in plt.gca().containers:
        plt.gca().bar_label(container, fmt='%d')

    plt.tight_layout()
    plt.savefig("wsb_sentiment_comparison.png")

    # Visualize WSB-specific terms and emojis
    plt.figure(figsize=(12, 12))

    # Top WSB terms
    plt.subplot(2, 2, 1)
    terms = [term for term, count in wsb_report['top_wsb_terms'][:10]]
    counts = [count for term, count in wsb_report['top_wsb_terms'][:10]]
    sns.barplot(x=counts, y=terms)
    plt.title("Top WSB Terms")
    plt.xlabel("Count")

    # Top Emojis
    plt.subplot(2, 2, 2)
    emojis = [emoji for emoji, count in wsb_report['top_emojis'][:10]]
    counts = [count for emoji, count in wsb_report['top_emojis'][:10]]
    sns.barplot(x=counts, y=emojis)
    plt.title("Top Emojis")
    plt.xlabel("Count")

    # Top Tickers
    plt.subplot(2, 2, 3)
    tickers = [f"${ticker}" for ticker, count in wsb_report['top_tickers'][:10]]
    counts = [count for ticker, count in wsb_report['top_tickers'][:10]]
    sns.barplot(x=counts, y=tickers)
    plt.title("Top Mentioned Tickers")
    plt.xlabel("Count")

    # Sentiment distribution
    plt.subplot(2, 2, 4)
    sentiment_data = pd.DataFrame({
        'Sentiment': ['Positive', 'Neutral', 'Negative'],
        'Count': [
            blended_counts.get('positive', 0),
            blended_counts.get('neutral', 0),
            blended_counts.get('negative', 0)
        ]
    })
    sns.barplot(x='Sentiment', y='Count', data=sentiment_data, palette={"Positive": "green", "Neutral": "gray", "Negative": "red"})
    plt.title("Blended Sentiment Distribution")

    plt.tight_layout()
    plt.savefig("wsb_language_analysis.png")

    # Find top posts for each sentiment
    top_positive = df[df['sentiment'] == 'positive'].sort_values('sentiment_score', ascending=False).head(5)
    top_negative = df[df['sentiment'] == 'negative'].sort_values('sentiment_score', ascending=False).head(5)

    print("\nTop Positive Posts:")
    for idx, post in top_positive.iterrows():
        print(f"- {post['title']} (Score: {post['score']}, WSB Terms: {', '.join(post['wsb_terms'][:3])}, Emojis: {', '.join(post['emojis'][:3])})")

    print("\nTop Negative Posts:")
    for idx, post in top_negative.iterrows():
        print(f"- {post['title']} (Score: {post['score']}, WSB Terms: {', '.join(post['wsb_terms'][:3])}, Emojis: {', '.join(post['emojis'][:3])})")

    # Create a ticker sentiment analysis
    print("\nAnalyzing ticker-specific sentiment...")
    ticker_sentiment = defaultdict(list)

    for _, row in df.iterrows():
        for ticker in row['tickers']:
            ticker_sentiment[ticker].append({
                'sentiment': row['sentiment'],
                'score': row['sentiment_score']
            })

    # Calculate average sentiment for each ticker
    ticker_avg_sentiment = {}
    for ticker, sentiments in ticker_sentiment.items():
        if len(sentiments) >= 3:  # Only consider tickers mentioned at least 3 times
            # Convert sentiment to numerical for averaging
            numerical_scores = []
            for s in sentiments:
                if s['sentiment'] == 'positive':
                    numerical_scores.append(s['score'])
                elif s['sentiment'] == 'negative':
                    numerical_scores.append(-s['score'])
                else:
                    numerical_scores.append(0)

            avg_score = sum(numerical_scores) / len(numerical_scores)
            ticker_avg_sentiment[ticker] = {
                'avg_score': avg_score,
                'mentions': len(sentiments)
            }

    # Sort tickers by average sentiment
    sorted_tickers = sorted(ticker_avg_sentiment.items(), key=lambda x: x[1]['avg_score'], reverse=True)

    print("\nTicker Sentiment Analysis (min 3 mentions):")
    print("Most Bullish Tickers:")
    for ticker, data in sorted_tickers[:20]:
        print(f"${ticker}: Score={data['avg_score']:.2f}, Mentions={data['mentions']}")

    print("\nMost Bearish Tickers:")
    for ticker, data in sorted_tickers[-20:]:
        print(f"${ticker}: Score={data['avg_score']:.2f}, Mentions={data['mentions']}")

    # Save results
    df.to_csv("wsb_enhanced_analysis.csv", index=False)
    print("\nAnalysis complete! Results saved to wsb_enhanced_analysis.csv")

    return df

# Run the analysis (adjust parameters as needed)
if __name__ == "__main__":
    results_df = analyze_wsb_sentiment(limit=10000, time_filter="year")
