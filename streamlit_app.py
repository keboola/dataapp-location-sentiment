import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import plotly.express as px

from openai import OpenAI
from wordcloud import WordCloud

OPENAI_API_KEY = st.secrets.get('OPENAI_API_KEY')
SENTIMENT_PATH = st.secrets['SENTIMENT_PATH']
KEYWORDS_PATH = st.secrets['KEYWORDS_PATH']
LOGO_URL = 'https://assets-global.website-files.com/5e21dc6f4c5acf29c35bb32c/5e21e66410e34945f7f25add_Keboola_logo.svg'


def display_logo():
    st.markdown(
        f'''
        <div style="text-align: right;">
            <img src="{LOGO_URL}" alt="Logo" width="150">
        </div>
        ''',
        unsafe_allow_html=True
    )

def color_for_value(value):
    if value < -0.2:
        return '#EA4335'
    elif -0.2 <= value <= 0.2:
        return '#FBBC05' 
    else:
        return '#34A853' 

def sentiment_color(sentiment):
    if sentiment == 'Positive':
        return 'color: #34A853'
    if sentiment == 'Neutral':
        return 'color: #FBBC05'
    else:
        return 'color: #EA4335'

def generate(content):
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        
        completion = client.chat.completions.create(
            model='gpt-4o-mini',
            messages=[
                {'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': content}
            ]
        )
        return completion.choices[0].message.content
    except Exception as e:
        st.error(f'An error occurred during content generation. Please try again.')
        return None

def generate_wordcloud(word_freq):
    colormap = mcolors.ListedColormap(['#4285F4', '#34A853', '#FBBC05', '#EA4335'])
    
    wordcloud = WordCloud(
        width=500, height=500, background_color=None, 
        colormap=colormap, mode='RGBA').generate_from_frequencies(word_freq)
    
    wordcloud_array = wordcloud.to_array()
    if wordcloud_array.dtype != np.uint8:
        wordcloud_array = wordcloud_array.astype(np.uint8)
    
    return wordcloud_array

def main():
    st.set_page_config(layout='wide')

    display_logo()

    st.title('Location Reviews Sentiment Analysis')

    if 'generated_responses' not in st.session_state:
        st.session_state.generated_responses = {}

    data = pd.read_csv(SENTIMENT_PATH)
    data['parsed_date'] = pd.to_datetime(data['parsed_date'], format='mixed').dt.tz_localize(None)
    data['date'] = data['parsed_date'].dt.date

    keywords = pd.read_csv(KEYWORDS_PATH)
    keywords['parsed_date'] = pd.to_datetime(keywords['parsed_date'], format='mixed').dt.tz_localize(None)
    keywords['date'] = keywords['parsed_date'].dt.date

    data['is_widget'] = False
    
    st.markdown('<br>__Filters__', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3, gap='medium')
    st.markdown('<br>', unsafe_allow_html=True)    
    with col1:
        min_score, max_score = st.slider(
            'Select a range for the sentiment score:',
            min_value=-1.0, max_value=1.0, value=(-1.0, 1.0),
            key='sentiment_slider'
        )

    with col2:
        source_choices = data['reviewSource'].unique().tolist()
        sources = st.multiselect('Filter by review source:', source_choices, placeholder='All')
        if len(sources) > 0:
            selected_source = sources
        else:
            selected_source = source_choices

    with col3:
        if not data.empty:
            min_date = data['parsed_date'].min()
            max_date = data['parsed_date'].max()
            default_date_range = (min_date, max_date)
        else:
            min_date, max_date = None, None
            default_date_range = ()

        date_range = st.date_input('Select a date range:', default_date_range, min_value=min_date, max_value=max_date)

    if date_range and len(date_range) == 2:
        start_date, end_date = date_range
        data = data[(data['parsed_date'] >= pd.to_datetime(start_date)) & (data['parsed_date'] <= pd.to_datetime(end_date))]
        keywords_filtered = keywords[(keywords['parsed_date'] >= pd.to_datetime(start_date)) & (keywords['parsed_date'] <= pd.to_datetime(end_date))]
    else:
        st.info('Please select both start and end dates.')
        st.stop()
        
    # Apply Filters
    filtered_data = data[(data['sentiment'] >= min_score) & (data['sentiment'] <= max_score) & (data['reviewSource'].isin(selected_source))]
    keywords_filtered = keywords_filtered[(keywords_filtered['sentiment'] >= min_score) & (keywords_filtered['sentiment'] <= max_score) & (keywords_filtered['reviewSource'].isin(selected_source))]

    col1, col2, col3 = st.columns(3, gap='medium')
    with col1:
        filtered_data['color'] = filtered_data['sentiment'].apply(color_for_value)
        fig = px.histogram(
            filtered_data,
            x='sentiment',
            nbins=21,  
            title='Sentiment Score Distribution',
            color='color',
            color_discrete_map='identity'  
        )

        fig.update_layout(bargap=0.1, xaxis_title='Sentiment Score', yaxis_title='Count') 
        st.plotly_chart(fig, use_container_width=True)

    with col2: 
        keyword_counts = keywords_filtered.groupby('keywords')['counts'].sum().reset_index()
        top_keywords = keyword_counts.sort_values(by='counts', ascending=True).tail(10)

        fig = px.bar(top_keywords, x='counts', y='keywords', orientation='h', title='Top 10 Keywords by Count', color_discrete_sequence=['#4285F4'])
        fig.update_layout(xaxis_title='Count', yaxis_title='Keywords')

        st.plotly_chart(fig, use_container_width=True)

    with col3:
        review_source_counts  = filtered_data['reviewSource'].value_counts().reset_index()
        review_source_counts .columns = ['reviewSource', 'count']
        top_10_industries = review_source_counts.head(10)
        count = top_10_industries.shape[0]
        colors = ['#63C5DA', '#0492C2', '#1F456E']
        
        fig = px.pie(
            top_10_industries, 
            names='reviewSource', 
            values='count', 
            title=f'Distribution of Reviews',
            color_discrete_sequence=colors
        )
        
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<br>', unsafe_allow_html=True)
    col1, col2 = st.columns([3,2])

    with col1:
        st.markdown('__Reviews__')
        sorted_data = filtered_data.sort_values(by='parsed_date', ascending=False)
        selected_data = st.data_editor(
            sorted_data[
                ['is_widget',
                'sentiment_category',
                'text_in_english',                            
                'stars',
                'reviewSource', 
                'date',
                'url']].style.map(sentiment_color, subset=["sentiment_category"]), 
                    column_config=
                    {'is_widget': 'Select',
                    'sentiment_category': 'Sentiment Category',
                    'text_in_english': 'Text',
                    'stars': 'Rating',
                    'reviewSource': 'Review Source',
                    'date': 'Date',
                    'url': st.column_config.LinkColumn('URL')}, 
                    height=500,
                    disabled=[
                        'sentiment_category',
                        'text_in_english',                            
                        'stars',
                        'reviewSource', 
                        'date',
                        'url'],
                    use_container_width=True, hide_index=True)
        
    summary = keywords_filtered.groupby('keywords')['counts'].sum().reset_index()
    word_freq = dict(zip(summary['keywords'], summary['counts']))

    # Wordcloud
    with col2:
        st.markdown('__Word Cloud__')
        if word_freq:    
            wordcloud = generate_wordcloud(word_freq)
            fig, ax = plt.subplots(figsize=(10, 5), frameon=False)
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')    
            st.pyplot(fig, use_container_width=True)
        else:
            st.info('No keywords found to generate the word cloud.')
            
    st.markdown('''
        <div style="text-align: left;">
            <h4>Reply to a review with AI</h4>
        </div>
        ''', 
        unsafe_allow_html=True)

    if selected_data['is_widget'].sum() == 1:
        selected_review = selected_data[selected_data['is_widget'] == True]['text_in_english'].iloc[0]
        review_text = selected_review if selected_review else st.warning('No review found.')
        st.write(f'_Review:_\n\n{review_text}')

        if st.button('Generate response'):
            if not OPENAI_API_KEY:
                st.warning('OpenAI API key is required to generate a response. Please provide the key.')
            else:
            
                if review_text in st.session_state['generated_responses']:
                    response = st.session_state['generated_responses'][review_text]
                else:
                    with st.spinner(':robot_face: Generating response, please wait...'):
                        prompt = f'''
                        Pretend you're a social media manager and write a short (3-5 sentence) response to the review you're given. Only return the response.
                        
                        Review:
                        {review_text}
                        '''
                        response = generate(prompt)
                        if response:
                            st.session_state['generated_responses'][review_text] = response
                            st.write(f'_Response:_\n\n{response}')
                        else:
                            st.error('Failed to generate a response. Please ensure your API key is correct and try again.')
    else:
        st.info('Select the review you want to respond to in the table above.')

if __name__ == '__main__':
    main()