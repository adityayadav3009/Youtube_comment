
import streamlit as st
import os
import pandas as pd
import googleapiclient.discovery
import nltk
import re
import numpy as np
from nltk.tokenize import word_tokenize
from string import punctuation
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
nltk.download('omw-1.4')
nltk.download('wordnet')
lemm = WordNetLemmatizer()
stop_nltk = stopwords.words("english")
stop_nltk.remove('not')
stop_updated = stop_nltk


def clean_txtlemma(sent):
    sent = sent.strip()
    sent = re.sub("🔥", "👌", sent)
    sent = re.sub("n't", "not", sent)
    sent = re.sub("not[\s]{1,}", "not_", sent)
    sent = re.sub("no[\s]{1,}", "not_", sent)
    tokens = word_tokenize(sent.lower())
    for i in range(len(tokens)):
        if tokens[i][:3] == "not":

            word = tokens[i][4:]
            antonyms = []
            for syn in wordnet.synsets(word):
                for l in syn.lemmas():
                    if l.antonyms():
                        antonyms.append(l.antonyms()[0].name())
            antonyms.sort()
            if len(antonyms) > 0:
                # if antonym is there then replace it , else ignore
                tokens[i] = antonyms[0]
    lemmatized = [lemm.lemmatize(
        term, pos='v') for term in tokens if term not in stop_updated and len(term) > 2]
    res = " ".join(lemmatized)
    return res


def comments(video_id):

    os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"

    api_service_name = "youtube"
    api_version = "v3"
    DEVELOPER_KEY = "AIzaSyBkGG5QXUqZJtXdZUbzFWpEPgsXgWoOudY"
    arr = []
    youtube = googleapiclient.discovery.build(
        api_service_name, api_version, developerKey=DEVELOPER_KEY)

    request = youtube.commentThreads().list(
        part=" snippet",
        maxResults=100,
        order="relevance",
        videoId=video_id
    )

    response = request.execute()

    page = 1
    for i in range(len(response["items"])):
        arr.append([response["items"][i]["snippet"]['topLevelComment']["snippet"]["authorDisplayName"], response["items"][i]["snippet"]
                    ['topLevelComment']["snippet"]["likeCount"], response["items"][i]["snippet"]['topLevelComment']["snippet"]["textOriginal"]])
    while True:
        if "nextPageToken" in response:
            page += 1
            request = youtube.commentThreads().list(
                part=" snippet",
                maxResults=100,
                order="relevance",
                pageToken=response["nextPageToken"],
                videoId=video_id
            )

            response = request.execute()

            for i in range(len(response["items"])):
                arr.append([response["items"][i]["snippet"]['topLevelComment']["snippet"]["authorDisplayName"], response["items"][i]["snippet"]
                            ['topLevelComment']["snippet"]["likeCount"], response["items"][i]["snippet"]['topLevelComment']["snippet"]["textOriginal"]])
        else:
            break
        # if len(arr) >= 2000 :
        #      break

    return pd.DataFrame(arr, columns=["Name", "Likes", "Comment"])


def get_video_id(link):
    if len(link) == 0:
        return " none , Try pasting this =>  https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    return link.split("=")[1].split("&")[0]


st.set_page_config(
    page_title="YT Comment analyzer", page_icon="symbol.png", layout="centered"
)
st.title("Youtube Comment analyzer")
link = st.text_input(
    "Link of video")

video_id = get_video_id(link)
st.write("Video id :  {}".format(video_id))
if len(video_id) < 20:
    with st.spinner("Getting Comments"):
        df = comments(video_id)
    if len(df) == 0:
        st.error("No comments found")
    elif len(df) < 20:
        st.warning("Only {} comments found ".format(len(df)))
    else:
        st.success("Extracted top {} comments ".format(len(df)))

        def convert_df(df):
            return df.to_csv().encode('utf-8')
        csv = convert_df(df)

        st.download_button(
            label="Download data as CSV",
            data=csv,
            file_name='Comments{}.csv'.format(video_id),
            mime='text/csv',
        )
        st.markdown("Making WordCloud")
        df['lem_comments'] = df.Comment.apply(clean_txtlemma)
        df['lem_comments'] = df.Comment.apply(clean_txtlemma)
        df["Compound_vader_score"] = df["lem_comments"].apply(
            lambda x: SentimentIntensityAnalyzer().polarity_scores(x)["compound"])
        df["Opinion_score"] = df["lem_comments"].apply(
            lambda x: TextBlob(x).sentiment.subjectivity)
        df["polarity_score"] = df["lem_comments"].apply(
            lambda x: TextBlob(x).sentiment.polarity)
        df["final_score"] = (df["polarity_score"]+df["Compound_vader_score"])/2
        df["is_question"] = df["Comment"].apply(lambda x: 1 if "?" in x else 0)
        reviews_combined = " ".join(df["Comment"])
        fig, ax = plt.subplots(1, 1)
        word_cloud = WordCloud(width=800, height=800, background_color='white',
                               random_state=500).generate_from_text(reviews_combined)
        ax.imshow(word_cloud)
        ax.axis("off")
        st.pyplot(fig)
        fig, ax = plt.subplots(1, 1)
        sns.kdeplot(df["final_score"], ax=ax)
        ax.yaxis.set_visible(False)
        ax.set_xlabel("Sentiment")
        st.pyplot(fig)

