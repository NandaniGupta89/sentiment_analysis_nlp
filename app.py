import streamlit as st
from textblob import TextBlob
import pandas as pd
import altair as alt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import numpy as np
import joblib
import pickle
import time
import plotly.express as px
from datetime import datetime
import cleantext
def convert_to_df(sentiment):
	sentiment_dict = {'polarity':sentiment.polarity,'subjectivity':sentiment.subjectivity}
	sentiment_df = pd.DataFrame(sentiment_dict.items(),columns=['metric','value'])
	return sentiment_df

def analyze_token_sentiment(docx):
	analyzer = SentimentIntensityAnalyzer()
	pos_list = []
	neg_list = []
	neu_list = []
	for i in docx.split():
		res = analyzer.polarity_scores(i)['compound']
		if res > 0.1:
			pos_list.append(i)
			pos_list.append(res)

		elif res <= -0.1:
			neg_list.append(i)
			neg_list.append(res)
		else:
			neu_list.append(i)

	result = {'positives':pos_list,'negatives':neg_list,'neutral':neu_list}
	return result 

pipe_lr = joblib.load(open("model/text_emotion.pkl", "rb"))

emotions_emoji_dict = {"anger": "😠", "disgust": "🤮", "fear": "😨😱", "happy": "🤗", "joy": "😂", "neutral": "😐","sad": "😔","sadness": "😔", "shame": "😳", "surprise": "😮"}

def predict_emotions(docx):
	results = pipe_lr.predict([docx])
	return results[0]

def get_prediction_proba(docx):
	results = pipe_lr.predict_proba([docx])
	return results

model = pickle.load(open('twitter_sentiment.pkl', 'rb'))

model = pickle.load(open('sentiment_analysis.pkl', 'rb'))

pipe_lr = joblib.load(open("notebooks/emotion_classifier_pipe_lr_03_june_2021.pkl","rb"))

from track_utils import create_page_visited_table,add_page_visited_details,view_all_page_visited_details,add_prediction_details,view_all_prediction_details,create_emotionclf_table,main

def main():

	menu = ["Home","Amazon","Emotion","Monitor","Twitter","Movie","About"]
	choice = st.sidebar.selectbox("Menu",menu)
	if choice == "Home":
		st.title("Sentiment Analysis")


		with st.expander('Analyze Text'):
			with st.form(key='nlpForm'):

				pre = st.text_input('Clean Text: ')
				submit_button = st.form_submit_button(label='submit')
			if submit_button:
				cleaned_text = cleantext.clean(pre, clean_all=False, extra_spaces=True,
											   stopwords=True, lowercase=True, numbers=True, punct=True)
				st.write(cleaned_text)



	elif choice == "Amazon":

		st.title("Amazon Product Review Analysis")

		with st.form(key='nlpForm'):
			raw_text = st.text_area("Enter Text Here")
			submit_button = st.form_submit_button(label='Analyze')

		# layout
		col1,col2 = st.columns(2)
		if submit_button:

			with col1:
				st.info("Results")
				sentiment = TextBlob(raw_text).sentiment
				st.write(sentiment)

				# Emoji
				if sentiment.polarity > 0:
					st.markdown("Sentiment:: Positive :smiley: ")
				elif sentiment.polarity < 0:
					st.markdown("Sentiment:: Negative :angry: ")
				else:
					st.markdown("Sentiment:: Neutral 😐 ")

				# Dataframe
				result_df = convert_to_df(sentiment)
				st.dataframe(result_df)

				# Visualization
				c = alt.Chart(result_df).mark_bar().encode(
					x='metric',
					y='value',
					color='metric')
				st.altair_chart(c,use_container_width=True)

			with col2:
				st.info("Token Sentiment")

				token_sentiments = analyze_token_sentiment(raw_text)
				st.write(token_sentiments)

	elif choice == "Emotion":
		add_page_visited_details("Home", datetime.now())
		st.title("Emotion Detection Analysis")


		with st.form(key='emotion_clf_form'):
			raw_text = st.text_area("Type Here")
			submit_text = st.form_submit_button(label='Submit')

		if submit_text:
			col1, col2 = st.columns(2)

			# Apply Fxn Here
			prediction = predict_emotions(raw_text)
			probability = get_prediction_proba(raw_text)

			add_prediction_details(raw_text, prediction, np.max(probability), datetime.now())

			with col1:
				st.success("Original Text")
				st.write(raw_text)

				st.success("Prediction")
				emoji_icon = emotions_emoji_dict[prediction]
				st.write("{}:{}".format(prediction, emoji_icon))
				st.write("Confidence:{}".format(np.max(probability)))

			with col2:
				st.success("Prediction Probability")
				# st.write(probability)
				proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_)
				# st.write(proba_df.T)
				proba_df_clean = proba_df.T.reset_index()
				proba_df_clean.columns = ["emotions", "probability"]

				fig = alt.Chart(proba_df_clean).mark_bar().encode(x='emotions', y='probability', color='emotions')
				st.altair_chart(fig, use_container_width=True)

	elif choice == "Monitor":
		add_page_visited_details("Monitor", datetime.now())
		st.title("Database Of Emotion")


		with st.expander("Page Metrics"):
			page_visited_details = pd.DataFrame(view_all_page_visited_details(), columns=['Pagename', 'Time_of_Visit'])
			st.dataframe(page_visited_details)

			pg_count = page_visited_details['Pagename'].value_counts().rename_axis('Pagename').reset_index(
				name='Counts')
			c = alt.Chart(pg_count).mark_bar().encode(x='Pagename', y='Counts', color='Pagename')
			st.altair_chart(c, use_container_width=True)

			p = px.pie(pg_count, values='Counts', names='Pagename')
			st.plotly_chart(p, use_container_width=True)

		with st.expander('Emotion Classifier Metrics'):
			df_emotions = pd.DataFrame(view_all_prediction_details(),
									   columns=['Rawtext', 'Prediction', 'Probability', 'Time_of_Visit'])
			st.dataframe(df_emotions)

			prediction_count = df_emotions['Prediction'].value_counts().rename_axis('Prediction').reset_index(
				name='Counts')
			pc = alt.Chart(prediction_count).mark_bar().encode(x='Prediction', y='Counts', color='Prediction')
			st.altair_chart(pc, use_container_width=True)

	elif choice == "Twitter":

		st.title('Twitter Sentiment Analysis')

		tweet = st.text_input('Enter your tweet')

		submit = st.button('Predict')

		if submit:
			start = time.time()
			prediction = model.predict([tweet])
			end = time.time()
			st.write('Prediction time taken: ', round(end - start, 2), 'seconds')

			print(prediction[0])
			st.write(prediction[0])
	elif choice == "Movie":

		st.title('Movie Review Analysis')

		review = st.text_input('Enter your review:')

		submit = st.button('Predict')

		if submit:
			prediction = model.predict([review])

			if prediction[0] == 'positive':
				st.success('Positive Review')
			else:
				st.warning('Negative Review')
	else:
		st.write("<h1 style='text-align: center;'>About</h1>", unsafe_allow_html=True)
		st.write(""" "Sentiment analysis, also known as opinion mining, is a Natural Language Processing (NLP) technique used to determine the sentiment or emotion expressed in a piece of text. It involves analyzing and categorizing text data to identify whether the sentiment conveyed is positive, negative, or neutral. Sentiment analysis has numerous applications across various industries, including social media monitoring(Twitter Tweets), customer feedback analysis(Movie Reviews), and brand sentiment tracking(Amazon Product)." """)
		st.write(""" A typical setting aims to categorize a text as positive, negative, or neutral.More fine-grained sentiment classification is possible by using a sentiment score, which can, for instance, range from -10 (very negative) to 0 (neutral) to +10 (very positive).""")
		st.write("""1. Clean text - The task of cleaning text in a sentence often involves preprocessing steps such as removing punctuation, converting text to lowercase, removing stopwords (commonly used words like "the", "is", "and", etc.), and possibly stemming or lemmatizing words (reducing them to their base or dictionary form).""")
		st.write("2. Polarity - Refers to the degree of positivity or negativity in a given text. In NLP, polarity analysis is used to determine the sentiment of a text, whether it is positive, negative, or neutral")
		st.write("3. Subjectivity - Refers to the degree of personal opinion or bias in a given text. In NLP, subjectivity analysis is used to determine whether a text is objective or subjective.")
if __name__ == '__main__':
	main()



