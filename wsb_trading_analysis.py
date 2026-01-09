"""
WSB Trading Analysis Module

This module uses FinBERT's capabilities to analyze WSB data and answer questions about
trading activities mentioned in the posts. It provides insights into what kinds of trades
people are placing on WallStreetBets.

Requirements:
- transformers
- pandas
- torch
- nltk
- matplotlib
- seaborn

Example Usage:
```python
# Initialize the analyzer with the WSB CSV file
from wsb_trading_analysis import WSBTradingAnalyzer

# Create the analyzer
analyzer = WSBTradingAnalyzer('wsb_enhanced_analysis.csv')

# Run comprehensive analysis
results = analyzer.run_comprehensive_analysis()
print(results['summary'])

# Ask specific questions
print(analyzer.answer_question("What are the most popular call options?"))
print(analyzer.answer_question("Are people bullish or bearish on NVDA?"))
print(analyzer.answer_question("What kind of trades are people placing on RKT?"))

# Analyze a specific ticker
tsla_analysis = analyzer.analyze_ticker('TSLA')
print(tsla_analysis['summary'])

# Generate visualizations
analyzer.plot_sentiment_distribution()
analyzer.plot_trading_actions()
analyzer.plot_top_tickers()
```
"""

import pandas as pd
import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
from transformers import BertForSequenceClassification, BertTokenizer
import re
import json
import nltk
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
import numpy as np

# Download necessary NLTK data
nltk.download('punkt', quiet=True)

class WSBTradingAnalyzer:
    def __init__(self, csv_path):
        """
        Initialize the WSB Trading Analyzer with a CSV file containing WSB posts

        Args:
            csv_path (str): Path to the CSV file with WSB post data
        """
        self.df = pd.read_csv(csv_path)

        # Clean and preprocess data
        self._preprocess_data()

        # Load FinBERT model for financial sentiment analysis
        self.finbert_model_name = "yiyanghkust/finbert-tone"
        self.finbert_tokenizer = BertTokenizer.from_pretrained(self.finbert_model_name)
        self.finbert_model = BertForSequenceClassification.from_pretrained(self.finbert_model_name)

        # Load question answering model
        self.qa_model_name = "deepset/roberta-base-squad2"
        self.qa_nlp = pipeline('question-answering', model=self.qa_model_name)

        # Extract trading actions from posts
        self._extract_trading_actions()

        # Trading-related terms
        self.trading_terms = [
            'call', 'calls', 'put', 'puts', 'long', 'short', 'buy', 'sell',
            'yolo', 'bet', 'options', 'leverage', 'stocks', 'shares', 'moon',
            'bullish', 'bearish', 'tendies'
        ]

    def _preprocess_data(self):
        """Clean and preprocess the data"""
        # Convert dates to datetime
        if 'created_utc' in self.df.columns:
            self.df['created_utc'] = pd.to_datetime(self.df['created_utc'])

        # Handle NaN values
        self.df['selftext'] = self.df['selftext'].fillna('')
        self.df['full_text'] = self.df['full_text'].fillna('')

        # Parse ticker lists that are stored as strings
        self.df['parsed_tickers'] = self.df['tickers'].apply(self._parse_list_string)

        # Parse WSB terms
        self.df['parsed_wsb_terms'] = self.df['wsb_terms'].apply(self._parse_list_string)

    def _parse_list_string(self, list_str):
        """Parse a string representation of a list into an actual list"""
        if pd.isna(list_str) or list_str == '[]':
            return []

        try:
            # Try parsing as JSON
            return json.loads(list_str)
        except:
            # Try parsing manually
            list_str = list_str.strip('[]').replace("'", "").replace('"', '')
            if not list_str:
                return []
            return [item.strip() for item in list_str.split(',')]

    def _extract_trading_actions(self):
        """Extract trading actions from post texts"""
        self.df['trading_action'] = self.df['full_text'].apply(self._identify_trading_action)
        self.df['has_trading_signal'] = ~self.df['trading_action'].isin(['UNKNOWN', 'NONE'])

    def _identify_trading_action(self, text):
        """
        Identify trading action from text

        Returns one of: CALL_OPTIONS, PUT_OPTIONS, BUY_SHARES, SELL_SHARES, HOLD, SHORT, LONG, UNKNOWN
        """
        if pd.isna(text):
            return 'NONE'

        text = text.lower()

        # Look for specific trading patterns
        if re.search(r'\bbuy\s+calls\b|\bcalls\s+on\b|\bcall\s+option', text) and not re.search(r'put', text):
            return 'CALL_OPTIONS'
        elif re.search(r'\bbuy\s+puts\b|\bputs\s+on\b|\bput\s+option', text) and not re.search(r'call', text):
            return 'PUT_OPTIONS'
        elif re.search(r'\bbuy\b', text) and not re.search(r'\bsell\b', text):
            return 'BUY_SHARES'
        elif re.search(r'\bsell\b', text) and not re.search(r'\bbuy\b', text):
            return 'SELL_SHARES'
        elif re.search(r'\bhold\b|\bhodl\b', text):
            return 'HOLD'
        elif re.search(r'\bshort\b', text):
            return 'SHORT'
        elif re.search(r'\blong\b', text):
            return 'LONG'

        return 'UNKNOWN'

    def answer_question(self, question):
        """
        Use QA model to answer a question about the WSB trading data

        Args:
            question (str): Question about WSB trading activities

        Returns:
            dict: Answer with supporting information
        """
        # Common questions mapping
        common_questions = {
            'top_tickers': self.get_top_tickers,
            'sentiment': self.get_sentiment_distribution,
            'trading_actions': self.get_trading_actions_distribution,
            'popular_options': self.get_popular_options_trades,
            'bullish_tickers': self.get_bullish_tickers,
            'bearish_tickers': self.get_bearish_tickers,
            'wsb_terms': self.get_popular_wsb_terms,
            'high_engagement': self.get_high_engagement_posts,
        }

        # Check for direct question matches
        question_lower = question.lower()

        if 'top' in question_lower and ('ticker' in question_lower or 'stock' in question_lower):
            return common_questions['top_tickers']()

        elif 'sentiment' in question_lower:
            return common_questions['sentiment']()

        elif 'trading' in question_lower and ('action' in question_lower or 'activity' in question_lower):
            return common_questions['trading_actions']()

        elif 'option' in question_lower:
            return common_questions['popular_options']()

        elif ('bullish' in question_lower or 'positive' in question_lower) and ('ticker' in question_lower or 'stock' in question_lower):
            return common_questions['bullish_tickers']()

        elif ('bearish' in question_lower or 'negative' in question_lower) and ('ticker' in question_lower or 'stock' in question_lower):
            return common_questions['bearish_tickers']()

        elif 'term' in question_lower or 'jargon' in question_lower or 'lingo' in question_lower:
            return common_questions['wsb_terms']()

        elif 'popular' in question_lower or 'engagement' in question_lower:
            return common_questions['high_engagement']()

        # If no direct match, try to find relevant posts
        relevant_posts = self._find_relevant_posts(question)

        # Use the QA model to extract answers from relevant posts
        answers = []
        for _, post in relevant_posts.iterrows():
            context = post['full_text']
            if len(context) > 10:  # Skip empty or very short posts
                try:
                    result = self.qa_nlp(question=question, context=context)
                    answers.append({
                        'answer': result['answer'],
                        'score': result['score'],
                        'post_title': post['title'],
                        'post_score': post['score'],
                        'trading_action': post['trading_action'],
                        'tickers': post['parsed_tickers'],
                        'sentiment': post['sentiment']
                    })
                except Exception as e:
                    continue

        # Filter answers with decent confidence
        good_answers = [ans for ans in answers if ans['score'] > 0.1]
        if good_answers:
            # Sort by confidence score
            good_answers.sort(key=lambda x: x['score'], reverse=True)
            return {
                'type': 'qa',
                'results': good_answers[:5],  # Return top 5 answers
                'summary': self._generate_answer_summary(good_answers[:5])
            }
        else:
            # Fall back to a default analysis
            return {
                'type': 'fallback',
                'message': "I couldn't find a specific answer to your question in the posts. Here's an overview of the trading activity instead:",
                'overview': {
                    'total_posts': len(self.df),
                    'posts_with_trading_signals': sum(self.df['has_trading_signal']),
                    'top_tickers': self.get_top_tickers()['tickers'][:5],
                    'sentiment_distribution': self.get_sentiment_distribution()['distribution']
                }
            }

    def _find_relevant_posts(self, question):
        """Find posts relevant to the question"""
        # Extract key terms from the question
        question_tokens = set(word_tokenize(question.lower()))

        # Check for ticker mentions
        tickers_mentioned = []
        for word in question_tokens:
            if word.startswith('$'):
                tickers_mentioned.append(word[1:].upper())
            elif word.isupper() and len(word) <= 5:
                tickers_mentioned.append(word)

        # Create a relevance score for each post
        self.df['relevance_score'] = 0

        # Increase score for posts that mention tickers in the question
        if tickers_mentioned:
            for ticker in tickers_mentioned:
                self.df['relevance_score'] += self.df['parsed_tickers'].apply(
                    lambda x: 3 if ticker in x else 0
                )

        # Check for trading terms in the question
        trading_terms_in_question = [term for term in self.trading_terms if term in question.lower()]

        # Increase score for posts that include the trading terms from the question
        for term in trading_terms_in_question:
            self.df['relevance_score'] += self.df['full_text'].apply(
                lambda x: 1 if pd.notna(x) and term.lower() in x.lower() else 0
            )

        # Add score for posts with trading actions
        if 'buy' in question.lower() or 'long' in question.lower() or 'call' in question.lower():
            self.df['relevance_score'] += self.df['trading_action'].apply(
                lambda x: 2 if x in ['BUY_SHARES', 'LONG', 'CALL_OPTIONS'] else 0
            )
        elif 'sell' in question.lower() or 'short' in question.lower() or 'put' in question.lower():
            self.df['relevance_score'] += self.df['trading_action'].apply(
                lambda x: 2 if x in ['SELL_SHARES', 'SHORT', 'PUT_OPTIONS'] else 0
            )

        # Add score for sentiment match
        if 'bullish' in question.lower() or 'positive' in question.lower():
            self.df['relevance_score'] += self.df['sentiment'].apply(
                lambda x: 1 if x == 'positive' else 0
            )
        elif 'bearish' in question.lower() or 'negative' in question.lower():
            self.df['relevance_score'] += self.df['sentiment'].apply(
                lambda x: 1 if x == 'negative' else 0
            )

        # Get top 10 most relevant posts
        relevant_posts = self.df.sort_values('relevance_score', ascending=False).head(10)
        return relevant_posts

    def _generate_answer_summary(self, answers):
        """Generate a summary of the answers"""
        if not answers:
            return "No relevant information found."

        # Extract key information
        tickers_mentioned = []
        sentiments = []
        actions = []

        for ans in answers:
            tickers_mentioned.extend(ans['tickers'])
            sentiments.append(ans['sentiment'])
            actions.append(ans['trading_action'])

        # Count occurrences
        ticker_counts = Counter(tickers_mentioned)
        sentiment_counts = Counter(sentiments)
        action_counts = Counter(actions)

        # Generate summary
        summary = "Based on the analyzed posts, "

        # Add ticker information
        if ticker_counts:
            most_common_tickers = ticker_counts.most_common(3)
            ticker_str = ", ".join([f"${t[0]} ({t[1]} mentions)" for t in most_common_tickers])
            summary += f"the most discussed stocks are {ticker_str}. "

        # Add sentiment information
        most_common_sentiment = max(sentiment_counts.items(), key=lambda x: x[1])[0] if sentiment_counts else None
        if most_common_sentiment:
            summary += f"The overall sentiment is {most_common_sentiment}. "

        # Add trading action information
        most_common_action = max(action_counts.items(), key=lambda x: x[1])[0] if action_counts else None
        if most_common_action != 'UNKNOWN' and most_common_action:
            action_desc = {
                'BUY_SHARES': 'buying shares',
                'SELL_SHARES': 'selling shares',
                'CALL_OPTIONS': 'buying call options',
                'PUT_OPTIONS': 'buying put options',
                'HOLD': 'holding positions',
                'SHORT': 'short selling',
                'LONG': 'going long'
            }.get(most_common_action, most_common_action)
            summary += f"Most users appear to be {action_desc}."

        return summary

    def get_top_tickers(self, n=20):
        """Get the top n most mentioned tickers"""
        # Flatten the list of tickers
        all_tickers = []
        for tickers in self.df['parsed_tickers']:
            all_tickers.extend(tickers)

        # Count occurrences
        ticker_counts = Counter(all_tickers)
        top_tickers = ticker_counts.most_common(n)

        return {
            'type': 'top_tickers',
            'tickers': [(ticker, count) for ticker, count in top_tickers],
            'summary': f"The most mentioned ticker is ${top_tickers[0][0]} with {top_tickers[0][1]} mentions, followed by ${top_tickers[1][0]} with {top_tickers[1][1]} mentions."
        }

    def get_sentiment_distribution(self):
        """Get the distribution of sentiment across posts"""
        sentiment_counts = self.df['sentiment'].value_counts().to_dict()
        total = sum(sentiment_counts.values())

        return {
            'type': 'sentiment_distribution',
            'distribution': sentiment_counts,
            'percentages': {k: round(v/total*100, 1) for k, v in sentiment_counts.items()},
            'summary': f"Out of {total} posts, {sentiment_counts.get('negative', 0)} are negative, {sentiment_counts.get('positive', 0)} are positive, and {sentiment_counts.get('neutral', 0)} are neutral."
        }

    def get_trading_actions_distribution(self):
        """Get the distribution of trading actions across posts"""
        action_counts = self.df['trading_action'].value_counts().to_dict()

        # Remove UNKNOWN and NONE from the counts
        if 'UNKNOWN' in action_counts:
            del action_counts['UNKNOWN']
        if 'NONE' in action_counts:
            del action_counts['NONE']

        total = sum(action_counts.values())

        # Map action names to more readable descriptions
        action_descriptions = {
            'BUY_SHARES': 'Buying shares',
            'SELL_SHARES': 'Selling shares',
            'CALL_OPTIONS': 'Call options',
            'PUT_OPTIONS': 'Put options',
            'HOLD': 'Holding positions',
            'SHORT': 'Short selling',
            'LONG': 'Going long'
        }

        readable_counts = {action_descriptions.get(k, k): v for k, v in action_counts.items()}

        return {
            'type': 'trading_actions',
            'distribution': action_counts,
            'readable_distribution': readable_counts,
            'percentages': {k: round(v/total*100, 1) for k, v in action_counts.items()},
            'summary': f"The most common trading action is {max(action_counts.items(), key=lambda x: x[1])[0]} with {max(action_counts.values())} mentions."
        }

    def get_popular_options_trades(self):
        """Get the most popular options trades"""
        # Filter for call and put options
        options_df = self.df[self.df['trading_action'].isin(['CALL_OPTIONS', 'PUT_OPTIONS'])]

        # Get ticker counts for each type
        call_options = options_df[options_df['trading_action'] == 'CALL_OPTIONS']
        put_options = options_df[options_df['trading_action'] == 'PUT_OPTIONS']

        call_tickers = []
        for tickers in call_options['parsed_tickers']:
            call_tickers.extend(tickers)

        put_tickers = []
        for tickers in put_options['parsed_tickers']:
            put_tickers.extend(tickers)

        call_counts = Counter(call_tickers).most_common(5)
        put_counts = Counter(put_tickers).most_common(5)

        return {
            'type': 'options_trades',
            'call_options': {ticker: count for ticker, count in call_counts},
            'put_options': {ticker: count for ticker, count in put_counts},
            'call_total': len(call_options),
            'put_total': len(put_options),
            'summary': f"There are {len(call_options)} posts about call options and {len(put_options)} posts about put options. "
                      f"The most popular ticker for calls is ${call_counts[0][0] if call_counts else 'N/A'} and for puts is ${put_counts[0][0] if put_counts else 'N/A'}."
        }

    def get_bullish_tickers(self):
        """Get tickers with positive sentiment"""
        # Filter for positive sentiment posts
        positive_df = self.df[self.df['sentiment'] == 'positive']

        # Get all tickers mentioned in positive posts
        positive_tickers = []
        for tickers in positive_df['parsed_tickers']:
            positive_tickers.extend(tickers)

        ticker_counts = Counter(positive_tickers).most_common(10)

        return {
            'type': 'bullish_tickers',
            'tickers': {ticker: count for ticker, count in ticker_counts},
            'total_positive_posts': len(positive_df),
            'summary': f"Out of {len(positive_df)} posts with positive sentiment, "
                      f"the most bullish ticker is ${ticker_counts[0][0] if ticker_counts else 'N/A'} with {ticker_counts[0][1] if ticker_counts else 0} mentions."
        }

    def get_bearish_tickers(self):
        """Get tickers with negative sentiment"""
        # Filter for negative sentiment posts
        negative_df = self.df[self.df['sentiment'] == 'negative']

        # Get all tickers mentioned in negative posts
        negative_tickers = []
        for tickers in negative_df['parsed_tickers']:
            negative_tickers.extend(tickers)

        ticker_counts = Counter(negative_tickers).most_common(10)

        return {
            'type': 'bearish_tickers',
            'tickers': {ticker: count for ticker, count in ticker_counts},
            'total_negative_posts': len(negative_df),
            'summary': f"Out of {len(negative_df)} posts with negative sentiment, "
                      f"the most bearish ticker is ${ticker_counts[0][0] if ticker_counts else 'N/A'} with {ticker_counts[0][1] if ticker_counts else 0} mentions."
        }

    def get_popular_wsb_terms(self):
        """Get the most popular WSB-specific terms"""
        # Flatten the list of WSB terms
        all_terms = []
        for terms in self.df['parsed_wsb_terms']:
            all_terms.extend(terms)

        # Count occurrences
        term_counts = Counter(all_terms).most_common(15)

        return {
            'type': 'wsb_terms',
            'terms': {term: count for term, count in term_counts},
            'summary': f"The most common WSB terms are: {', '.join([f'{term} ({count})' for term, count in term_counts[:5]])}"
        }

    def get_high_engagement_posts(self):
        """Get posts with high engagement (high score and comment count)"""
        # Sort by score (Reddit upvotes)
        high_score_posts = self.df.sort_values('score', ascending=False).head(10)

        # Get posts with trading signals
        trading_posts = high_score_posts[high_score_posts['has_trading_signal']]

        results = []
        for _, post in trading_posts.iterrows():
            tickers_str = ', '.join([f"${t}" for t in post['parsed_tickers']]) if post['parsed_tickers'] else 'None'

            results.append({
                'title': post['title'],
                'score': post['score'],
                'trading_action': post['trading_action'],
                'tickers': post['parsed_tickers'],
                'tickers_str': tickers_str,
                'sentiment': post['sentiment']
            })

        return {
            'type': 'high_engagement',
            'posts': results[:5],
            'summary': f"The highest engagement post is about {results[0]['tickers_str'] if results else 'N/A'} with {results[0]['score'] if results else 0} upvotes."
        }

    def analyze_ticker(self, ticker):
        """
        Analyze posts mentioning a specific ticker

        Args:
            ticker (str): Stock ticker symbol

        Returns:
            dict: Analysis results
        """
        # Standardize ticker format
        ticker = ticker.upper()

        # Filter posts mentioning this ticker
        ticker_posts = self.df[self.df['parsed_tickers'].apply(lambda x: ticker in x)]

        if len(ticker_posts) == 0:
            return {
                'type': 'ticker_analysis',
                'ticker': ticker,
                'found': False,
                'summary': f"No posts mentioning ${ticker} were found in the dataset."
            }

        # Count sentiment
        sentiment_counts = ticker_posts['sentiment'].value_counts().to_dict()

        # Count trading actions
        action_counts = ticker_posts['trading_action'].value_counts().to_dict()
        if 'UNKNOWN' in action_counts:
            del action_counts['UNKNOWN']
        if 'NONE' in action_counts:
            del action_counts['NONE']

        # Calculate average score
        avg_score = ticker_posts['score'].mean()

        # Get highest scored post
        top_post = ticker_posts.loc[ticker_posts['score'].idxmax()]

        # Determine overall sentiment
        if sentiment_counts.get('positive', 0) > sentiment_counts.get('negative', 0):
            overall_sentiment = 'bullish'
        elif sentiment_counts.get('negative', 0) > sentiment_counts.get('positive', 0):
            overall_sentiment = 'bearish'
        else:
            overall_sentiment = 'neutral'

        # Determine most common trading action
        most_common_action = max(action_counts.items(), key=lambda x: x[1])[0] if action_counts else 'UNKNOWN'

        return {
            'type': 'ticker_analysis',
            'ticker': ticker,
            'found': True,
            'mention_count': len(ticker_posts),
            'sentiment_distribution': sentiment_counts,
            'trading_actions': action_counts,
            'average_score': avg_score,
            'top_post': {
                'title': top_post['title'],
                'score': top_post['score'],
                'trading_action': top_post['trading_action'],
                'sentiment': top_post['sentiment']
            },
            'overall_sentiment': overall_sentiment,
            'most_common_action': most_common_action,
            'summary': f"${ticker} is mentioned in {len(ticker_posts)} posts with an overall {overall_sentiment} sentiment. "
                    f"The most common trading action for ${ticker} is {most_common_action}."
        }

    def finbert_qa(self, question, context=None):
        """
        Use FinBERT to answer financial questions about posts

        Args:
            question (str): Question to answer
            context (str, optional): Context to use. If None, uses the full dataset

        Returns:
            dict: FinBERT's answer
        """
        if context is None:
            # Sample relevant posts
            trading_posts = self.df[self.df['has_trading_signal']]
            if len(trading_posts) > 5:
                sample_posts = trading_posts.sample(5)
            else:
                sample_posts = trading_posts

            # Combine text from posts
            context = " ".join(sample_posts['full_text'].tolist())

        # Use QA pipeline
        result = self.qa_nlp(question=question, context=context)

        return {
            'type': 'finbert_qa',
            'question': question,
            'answer': result['answer'],
            'confidence': result['score'],
            'summary': f"Answer: {result['answer']} (confidence: {result['score']:.2f})"
        }

    def plot_sentiment_distribution(self):
        """Plot the distribution of sentiment"""
        plt.figure(figsize=(10, 6))
        sns.countplot(x='sentiment', data=self.df, palette={'positive': 'green', 'negative': 'red', 'neutral': 'gray'})
        plt.title('Sentiment Distribution in WSB Posts')
        plt.xlabel('Sentiment')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig('wsb_sentiment_distribution.png')
        plt.close()

        return 'wsb_sentiment_distribution.png'

    def plot_trading_actions(self):
        """Plot the distribution of trading actions"""
        # Filter out UNKNOWN and NONE
        trading_df = self.df[~self.df['trading_action'].isin(['UNKNOWN', 'NONE'])]

        plt.figure(figsize=(12, 6))
        sns.countplot(x='trading_action', data=trading_df)
        plt.title('Trading Actions in WSB Posts')
        plt.xlabel('Action')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('wsb_trading_actions.png')
        plt.close()

        return 'wsb_trading_actions.png'

    def plot_top_tickers(self, n=10):
        """Plot the top n tickers by mention count"""
        # Get ticker counts
        result = self.get_top_tickers(n)
        tickers = result['tickers']

        plt.figure(figsize=(12, 6))
        plt.bar([t[0] for t in tickers], [t[1] for t in tickers])
        plt.title(f'Top {n} Tickers Mentioned in WSB Posts')
        plt.xlabel('Ticker')
        plt.ylabel('Mention Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('wsb_top_tickers.png')
        plt.close()

        return 'wsb_top_tickers.png'

    def run_comprehensive_analysis(self):
        """Run a comprehensive analysis of the WSB data"""
        results = {
            'total_posts': len(self.df),
            'posts_with_trading_signals': sum(self.df['has_trading_signal']),
            'sentiment_distribution': self.get_sentiment_distribution(),
            'trading_actions': self.get_trading_actions_distribution(),
            'top_tickers': self.get_top_tickers(),
            'option_trades': self.get_popular_options_trades(),
            'bullish_tickers': self.get_bullish_tickers(),
            'bearish_tickers': self.get_bearish_tickers(),
            'wsb_terms': self.get_popular_wsb_terms(),
            'high_engagement': self.get_high_engagement_posts()
        }

        summary = f"""
        WSB Trading Analysis Summary:

        - Analyzed {results['total_posts']} posts, {results['posts_with_trading_signals']} with trading signals
        - Sentiment: {results['sentiment_distribution']['summary']}
        - Trading actions: {results['trading_actions']['summary']}
        - Top tickers: {results['top_tickers']['summary']}
        - Options: {results['option_trades']['summary']}
        - Bullish sentiment: {results['bullish_tickers']['summary']}
        - Bearish sentiment: {results['bearish_tickers']['summary']}
        - WSB terms: {results['wsb_terms']['summary']}
        - High engagement: {results['high_engagement']['summary']}
        """

        results['summary'] = summary
        return results
