# WallStreetBets (WSB) Sentiment & Trading Analysis

This project provides a comprehensive analytical framework for scraping, preprocessing, and analyzing financial discourse within the **r/WallStreetBets** subreddit. By integrating domain-specific linguistic rules with the **FinBERT** language model, the system extracts nuanced insights into retail investor behavior, ticker-specific sentiment, and high-risk trading strategies.

## 📁 Repository Structure

* **`WSB.pdf`**: Final coursework report detailing the community background, methodology, and comprehensive findings.
* **`integrated_wsb_analysis.py`**: The primary execution pipeline for data collection (via PRAW), preprocessing, and sentiment blending.
* **`wsb_preprocessing.py`**: A specialized module that handles WSB-specific slang (e.g., "tendies", "diamond hands") and emoji-based sentiment scoring.
* **`wsb_trading_analysis.py`**: Focuses on identifying specific trading actions (Calls, Puts, Buys, Sells) and provides a context-aware Question Answering (QA) interface.
* **`finbert_evaluation.py`**: A diagnostic toolkit to assess model performance using statistical metrics such as Bootstrap Confidence Intervals and McNemar's Test.
* **`wsb_analysis_code.py`**: Logic for generating analytical visualizations, including sentiment distributions, temporal trends, and correlation matrices.
* **`wsb_trading_demo.py`**: A demonstration script showcasing the analyzer's ability to process complex financial questions using natural language.

## 🚀 Key Features

### 1. Hybrid Sentiment Engine
The system utilizes a **blended sentiment approach** to account for the unique linguistic style of the WSB community:
* **FinBERT**: A specialized BERT model pre-trained on financial communications to understand market context.
* **WSB Dictionary**: A custom lexicon that assigns weights to community-specific terms (e.g., "moon" [+2.0] or "bagholder" [-1.5]).
* **Emoji Scoring**: Translates visual emotional markers (e.g., 🚀, 🐻) into numerical sentiment values.

### 2. Trading Action Identification
The system categorizes discourse into specific investment behaviors using regex-based pattern matching:
* **Options**: Identifies `CALL_OPTIONS` and `PUT_OPTIONS`.
* **Equities**: Categorizes actions as `BUY_SHARES`, `SELL_SHARES`, or `HOLD`.
* **Strategies**: Detects `LONG` or `SHORT` positions.

### 3. Analytics and Insights
* **Sentiment Bias**: The community exhibits a distinct "negative" bias (74.6%), frequently celebrating "loss porn" and risk-taking.
* **Content Optimization**: Maximum engagement is achieved through extremely concise titles (<50 characters) or deep-dive detailed analyses.
* **Temporal Patterns**: Peak activity occurs during evening hours (15:00-21:00 UTC), suggesting a predominantly North American user base.

## 📊 Summary of Findings
* **Top Ticker**: **Tesla (TSLA)** is the most discussed security, followed by emerging aerospace and semiconductor sectors (ACHR, NVDA).
* **Engagement**: A moderate positive correlation (0.39) exists between a post's score and its comment volume.
* **Emoji Usage**: Emojis are primarily used as emotional markers for negative outcomes (57.3% negative) rather than successes.

## 🛠️ Setup & Usage

1.  **Prerequisites**:
    * Reddit API Credentials (Client ID & Secret).
    * Libraries: `praw`, `pandas`, `transformers`, `torch`, `nltk`, `seaborn`, `matplotlib`, `emoji`.
2.  **Basic Usage**:
    ```python
    from wsb_trading_analysis import WSBTradingAnalyzer
    analyzer = WSBTradingAnalyzer('wsb_enhanced_analysis.csv')
    # Ask a targeted question
    print(analyzer.answer_question("What are the most popular call options?"))
    ```

## ⚖️ Statistical Validation
The `FinBERTEvaluator` ensures robustness through:
* **Calibration Assessment**: Reliability diagrams to ensure confidence scores match actual accuracy.
* **Bootstrap CI**: Estimating uncertainty through repeated sampling.
* **Cross-Validation**: Evaluating consistency across different data subsets to prevent overfitting.
