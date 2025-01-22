# Location Reviews Sentiment Analysis Dashboard

A Streamlit-based dashboard for analyzing sentiment and trends in location-based reviews. The application provides interactive visualizations and AI-powered review response generation capabilities.

## Features

- **Sentiment Analysis**: View sentiment score distribution across reviews
- **Keyword Analysis**: Explore top keywords and their frequencies
- **Interactive Filters**: Filter data by:
  - Sentiment score range
  - Review source
  - Date range
- **Visual Analytics**:
  - Sentiment score distribution histogram
  - Top keywords bar chart
  - Review source distribution pie chart
  - Interactive word cloud
- **AI-Powered Review Response**: Generate professional responses to reviews using GPT-4

## Setup

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Configure your secrets in `.streamlit/secrets.toml`:
   ```toml
   OPENAI_API_KEY = "your-api-key"
   SENTIMENT_PATH = "path/to/sentiment/data.csv"
   KEYWORDS_PATH = "path/to/keywords/data.csv"
   ```

## Running the Application

```bash
streamlit run streamlit_app.py
```

## Data Requirements

The application expects two CSV files:

### Sentiment Data
- Required columns:
  - `parsed_date`: Review date
  - `reviewSource`: Source of the review
  - `sentiment`: Numerical sentiment score (-1 to 1)
  - `sentiment_category`: Categorical sentiment (Positive/Neutral/Negative)
  - `text_in_english`: Review text
  - `stars`: Rating
  - `url`: Review URL

### Keywords Data
- Required columns:
  - `parsed_date`: Date
  - `keywords`: Extracted keywords
  - `counts`: Keyword frequency
  - `reviewSource`: Source of the review
  - `sentiment`: Associated sentiment score

## License

MIT License 