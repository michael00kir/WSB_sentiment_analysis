import re
import emoji
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from collections import Counter, defaultdict

class WSBPreprocessor:
    """
    A preprocessor class for Wall Street Bets language understanding.
    This handles WSB-specific terminology, emojis, and slang to improve sentiment analysis.
    """

    def __init__(self):
        # Download NLTK resources if needed
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')

        # WSB-specific terminology dictionary with sentiment adjustments
        # Positive value means positive sentiment, negative means negative
        self.wsb_terms = {
            # Positive terminology
            "tendies": 2.0,
            "moon": 2.0,
            "rocket": 2.0,
            "hodl": 1.5,
            "diamond hands": 1.5,
            "lambos": 1.5,
            "degenerates": 1.0,  # Positive in WSB context
            "apes": 1.0,
            "autists": 0.5,      # Often used affectionately
            "yolo": 1.0,
            "bullish": 1.5,
            "calls": 0.5,
            "dd": 0.5,           # Due Diligence is neutral/positive
            "stonks": 0.5,
            "gains": 1.5,
            "printing": 1.0,
            "brrrr": 1.0,

            # Negative terminology
            "bagholder": -1.5,
            "bears": -0.5,
            "bearish": -1.0,
            "puts": -0.5,
            "margin call": -1.5,
            "guh": -1.5,
            "loss porn": -0.5,   # Negative but celebrated in WSB
            "paper hands": -1.0,
            "short": -0.5,
            "drill": -1.0,
            "dump": -1.5,
            "crash": -1.5,
            "red": -1.0,
            "rope": -2.0,        # Suicide reference, very negative
            "losses": -1.5,
            "expired worthless": -1.5,
        }

        # Emoji sentiment dictionary
        self.emoji_sentiment = {
            "🚀": 2.0,           # Rocket = strong positive
            "🌙": 1.5,           # Moon = positive
            "💎": 1.5,           # Diamond = positive (diamond hands)
            "🙌": 1.0,           # Raised hands = positive
            "🦍": 1.0,           # Gorilla/ape = positive in WSB
            "🐂": 1.5,           # Bull = positive
            "📈": 1.5,           # Chart up = positive
            "💰": 1.5,           # Money bag = positive
            "💵": 1.0,           # Dollar = positive
            "🤑": 1.5,           # Money face = positive
            "🍗": 1.0,           # Poultry leg ("tendies") = positive
            "🐻": -1.0,          # Bear = negative in WSB
            "📉": -1.5,          # Chart down = negative
            "💩": -1.0,          # Poop = negative
            "🔥": 0.5,           # Fire = could be positive or negative (context dependent)
            "⚰️": -1.5,          # Coffin = negative
            "🪦": -1.5,          # Tombstone = negative
            "🧻": -1.0,          # Toilet paper ("paper hands") = negative
        }

        # Compile regex patterns for efficiency
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        self.ticker_pattern = re.compile(r'\$([A-Z]{1,5})\b')
        self.wsb_term_pattern = self._compile_wsb_pattern()

    def _compile_wsb_pattern(self):
        """Compile regex pattern for WSB terms"""
        # Sort terms by length (longest first) to catch phrases before individual words
        sorted_terms = sorted(self.wsb_terms.keys(), key=len, reverse=True)
        pattern = r'\b(?:' + '|'.join(re.escape(term) for term in sorted_terms) + r')\b'
        return re.compile(pattern, re.IGNORECASE)

    def preprocess(self, text):
        """Preprocess text by removing URLs, normalizing, etc."""
        if pd.isna(text) or text == "":
            return ""

        # Convert to lowercase
        text = text.lower()

        # Remove URLs
        text = self.url_pattern.sub('', text)

        # Replace common WSB-specific contractions and abbreviations
        text = text.replace('$tsla', 'tesla')
        text = text.replace('$gme', 'gamestop')
        text = text.replace('$amc', 'amc entertainment')

        return text

    def extract_features(self, text):
        """Extract WSB-specific features from the text"""
        if pd.isna(text) or text == "":
            return {
                'wsb_terms': [],
                'emojis': [],
                'tickers': [],
                'wsb_sentiment_score': 0,
                'emoji_sentiment_score': 0
            }

        # Extract tickers (e.g., $AAPL)
        tickers = self.ticker_pattern.findall(text)

        # Extract WSB terms
        wsb_terms_found = []
        for match in self.wsb_term_pattern.finditer(text.lower()):
            term = match.group(0)
            wsb_terms_found.append(term)

        # Extract emojis
        emojis_found = []
        emoji_sentiment_score = 0

        for char in text:
            if char in emoji.EMOJI_DATA:
                emojis_found.append(char)
                emoji_sentiment_score += self.emoji_sentiment.get(char, 0)

        # Calculate WSB-specific sentiment score
        wsb_sentiment_score = sum(self.wsb_terms.get(term.lower(), 0) for term in wsb_terms_found)

        return {
            'wsb_terms': wsb_terms_found,
            'emojis': emojis_found,
            'tickers': tickers,
            'wsb_sentiment_score': wsb_sentiment_score,
            'emoji_sentiment_score': emoji_sentiment_score
        }

    def analyze_post(self, text):
        """Analyze a single WSB post"""
        processed_text = self.preprocess(text)
        features = self.extract_features(text)

        # Combine sentiment scores
        total_wsb_sentiment = features['wsb_sentiment_score'] + features['emoji_sentiment_score']

        # Determine overall WSB-specific sentiment
        wsb_sentiment = "neutral"
        if total_wsb_sentiment > 1.5:
            wsb_sentiment = "positive"
        elif total_wsb_sentiment < -1.0:
            wsb_sentiment = "negative"

        return {
            'processed_text': processed_text,
            'wsb_terms': features['wsb_terms'],
            'emojis': features['emojis'],
            'tickers': features['tickers'],
            'wsb_sentiment': wsb_sentiment,
            'wsb_sentiment_score': total_wsb_sentiment
        }

    def process_dataframe(self, df, text_column='full_text'):
        """Process an entire dataframe of WSB posts"""
        results = []

        for text in df[text_column]:
            results.append(self.analyze_post(text))

        # Add new columns to the dataframe
        df['processed_text'] = [r['processed_text'] for r in results]
        df['wsb_terms'] = [r['wsb_terms'] for r in results]
        df['emojis'] = [r['emojis'] for r in results]
        df['tickers'] = [r['tickers'] for r in results]
        df['wsb_sentiment'] = [r['wsb_sentiment'] for r in results]
        df['wsb_sentiment_score'] = [r['wsb_sentiment_score'] for r in results]

        return df

    def get_top_wsb_terms(self, df, n=20):
        """Get the most common WSB terms used in the posts"""
        all_terms = []
        for terms in df['wsb_terms']:
            all_terms.extend(terms)

        return Counter(all_terms).most_common(n)

    def get_top_emojis(self, df, n=20):
        """Get the most common emojis used in the posts"""
        all_emojis = []
        for emojis in df['emojis']:
            all_emojis.extend(emojis)

        return Counter(all_emojis).most_common(n)

    def get_top_tickers(self, df, n=20):
        """Get the most mentioned tickers in the posts"""
        all_tickers = []
        for tickers in df['tickers']:
            all_tickers.extend(tickers)

        return Counter(all_tickers).most_common(n)

    def generate_summary_report(self, df):
        """Generate a summary report of WSB language analysis"""
        report = {
            'total_posts': len(df),
            'wsb_sentiment_distribution': df['wsb_sentiment'].value_counts().to_dict(),
            'top_wsb_terms': self.get_top_wsb_terms(df, 10),
            'top_emojis': self.get_top_emojis(df, 10),
            'top_tickers': self.get_top_tickers(df, 10),
            'avg_wsb_sentiment_score': df['wsb_sentiment_score'].mean()
        }

        return report

# Example usage:
if __name__ == "__main__":
    # Load saved data
    df = pd.read_csv("wsb_sentiment_analysis.csv")

    # Create and apply preprocessor
    preprocessor = WSBPreprocessor()
    processed_df = preprocessor.process_dataframe(df)

    # Generate and print report
    report = preprocessor.generate_summary_report(processed_df)
    print("WSB Language Analysis Report:")
    print(f"Total Posts: {report['total_posts']}")
    print(f"WSB Sentiment Distribution: {report['wsb_sentiment_distribution']}")
    print(f"Average WSB Sentiment Score: {report['avg_wsb_sentiment_score']:.2f}")
    print("\nTop WSB Terms:")
    for term, count in report['top_wsb_terms']:
        print(f"- {term}: {count}")

    print("\nTop Emojis:")
    for emoji_char, count in report['top_emojis']:
        print(f"- {emoji_char}: {count}")

    print("\nTop Tickers:")
    for ticker, count in report['top_tickers']:
        print(f"- {ticker}: {count}")

    # Save processed data
    processed_df.to_csv("wsb_processed_data.csv", index=False)
    print("\nProcessed data saved to wsb_processed_data.csv")
