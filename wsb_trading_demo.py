"""
WSB Trading Analysis Demo

This script demonstrates how to use the WSBTradingAnalyzer module to analyze
WallStreetBets trading activities using FinBERT's capabilities.
"""

from wsb_trading_analysis import WSBTradingAnalyzer
import json
import pprint
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

#Download required NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab', quiet=True)

def main():
    # Initialize the analyzer with the CSV file
    print("Initializing WSB Trading Analyzer...")
    analyzer = WSBTradingAnalyzer('wsb_enhanced_analysis.csv')

    # 1. Run a comprehensive analysis
    print("\n1. Running comprehensive analysis...")
    results = analyzer.run_comprehensive_analysis()
    print(results['summary'])

    # 2. Ask specific questions about trading activities
    print("\n2. Asking specific questions about trading activities...")

    questions = [
        "What are the most popular stocks on WSB?",
        "What kind of options trades are people making?",
        "Which tickers are people most bullish on?",
        "What are people saying about NVDA?",
        "Are people buying or selling RKT?",
        "What are the most common WSB terms used?",
        "What posts have the highest engagement?"
    ]

    for question in questions:
        print(f"\nQ: {question}")
        answer = analyzer.answer_question(question)
        print(f"A: {answer.get('summary', '')}")

    # 3. Analyze specific tickers
    print("\n3. Analyzing specific tickers...")
    tickers_to_analyze = ['TSLA', 'NVDA', 'RKT', 'PLTR', 'AMC']

    for ticker in tickers_to_analyze:
        analysis = analyzer.analyze_ticker(ticker)
        print(f"\n{ticker}: {analysis['summary']}")
        if analysis['found']:
            print(f"  Most common trading action: {analysis['most_common_action']}")
            print(f"  Sentiment distribution: {analysis['sentiment_distribution']}")

    # 4. Generate visualizations
    print("\n4. Generating visualizations...")
    sentiment_plot = analyzer.plot_sentiment_distribution()
    print(f"Sentiment distribution plot saved as: {sentiment_plot}")

    actions_plot = analyzer.plot_trading_actions()
    print(f"Trading actions plot saved as: {actions_plot}")

    tickers_plot = analyzer.plot_top_tickers()
    print(f"Top tickers plot saved as: {tickers_plot}")

    # 5. Perform FinBERT-specific QA
    print("\n5. Performing FinBERT-specific QA...")
    finbert_questions = [
        "What is the sentiment towards tech stocks?",
        "Are there any insider trading discussions?",
        "What options strategies are being used?",
        "What are the most popular stocks on WSB?",
        "What kind of options trades are people making?",
        "Which tickers are people most bullish on?",
        "What are people saying about NVDA?",
        "Are people buying or selling RKT?",
        "What are the most common WSB terms used?",
        "What posts have the highest engagement?",
        "What kind of investor behavior does Wall Street Bets exhibit, based on language and sentiment?",
        "Do WSB users show signs of fear or greed when discussing stocks like GameStop or AMC?",
        "How do posts in WSB reflect the ‘YOLO’ mentality or FOMO (Fear of Missing Out) when it comes to investing?",
        "What predictions are Wall Street Bets users making about the stock market in the short term?",
        "Do Wall Street Bets users often engage in speculative trading, and if so, what stocks are they targeting?",
        "How does WSB react to market events or earnings reports (e.g., quarterly results, Federal Reserve decisions)?",
        "How does the Wall Street Bets community react to sudden market changes, such as major stock price fluctuations or economic news?",
        "What emotions (fear, excitement, frustration, etc.) dominate the sentiment in WSB posts during a market crash or rally?",
        "What trends or shifts in tone can be identified when discussing massive price spikes like GameStop or AMC?",
        "What is the overall sentiment in Wall Street Bets posts regarding the impact of tariffs on U.S. companies?",
        "How do WSB users feel about recent tariff announcements or changes in trade policy?",
        "Is the sentiment about tariffs generally negative or positive in the context of U.S.-China trade relations?",
        "Do WSB posts express concern or optimism about tariffs imposed on foreign goods?",
        # Industries Affected by Tariffs
            "What industries are Wall Street Bets users most concerned about in relation to new tariffs?",
            "Which stocks in sectors like tech, automotive, or manufacturing are frequently discussed in connection with tariffs?",
            "Do WSB users mention any specific companies that might benefit from tariffs (e.g., domestic producers) or lose out due to them (e.g., import-heavy companies)?",

            # Trade War Sentiment
            "How does Wall Street Bets view the ongoing trade war between the U.S. and China in terms of stock market performance?",
            "Is there a bullish or bearish tone in WSB discussions about the economic effects of tariffs during periods of trade escalation?",
            "What phrases or key terms do WSB users associate with trade wars and tariffs—'market correction,' 'supply chain disruptions,' 'inflation,' etc.?",

            # Risk and Speculation on Tariffs
            "How often do WSB posts discuss the risk of tariffs affecting market volatility?",
            "Are there any specific high-risk trades or speculative strategies mentioned in relation to the uncertainty caused by tariffs?",
            "Do WSB users consider tariffs a short-term issue, or do they think their impact will have long-term effects on market performance?",

            # Investor Behavior During Tariff Announcements
            "What emotional tone (fear, excitement, confusion, etc.) is expressed in WSB posts right after major tariff announcements?",
            "Do WSB traders tend to react to tariffs with FOMO (fear of missing out) on certain stocks, or are they more cautious?",
            "What kind of trades (e.g., options, stocks) are popular on WSB when new tariffs are announced?",

            # Comparative Analysis of Tariff-Related Sentiment
            "How does the sentiment about tariffs in WSB compare to mainstream financial news sources?",
            "Are there noticeable differences in sentiment between discussions of tariffs affecting specific sectors, like tech versus manufacturing?",

            # Impact of Tariffs on Market Predictions
            "Do WSB users predict a downturn or a rally in the market due to the imposition of tariffs?",
            "How often do WSB posts speculate on the effects of tariffs on global supply chains and stock prices?",
            "What stocks or sectors do WSB users expect to outperform or underperform because of tariffs in the coming months?",

            # Tone Around Political Decisions on Tariffs
            "How does Wall Street Bets perceive political decisions around tariffs—are they discussed with skepticism, cynicism, or trust?",
            "Do WSB users feel that tariffs are being used as a tool for political leverage or economic protectionism?",

            # Tariffs and Inflation
            "What is the sentiment in Wall Street Bets about the potential for tariffs to lead to inflation or price increases?",
            "Are WSB users worried about the long-term effects of tariffs driving up consumer prices, or is the focus more on stock prices?"
    ]

    for question in finbert_questions:
        print(f"\nQ: {question}")
        answer = analyzer.finbert_qa(question)
        print(f"A: {answer['answer']} (confidence: {answer['confidence']:.2f})")

    print("\nWSB Trading Analysis Demo completed successfully!")

if __name__ == "__main__":
    main()
