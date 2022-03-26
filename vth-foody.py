#Import
#EDA
import os
from os import path
import json
import pkgutil
import re
# from tkinter import N
import numpy as np
import pandas as pd
import sqlite3 as sql
import difflib
import pandas_profiling as pp
from pandas_profiling import ProfileReport
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import io
import streamlit as st
from streamlit_pandas_profiling import st_profile_report
from PIL import Image
import wordcloud

import pickle

#NLP & ML
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel,cosine_similarity
from underthesea import word_tokenize,pos_tag,sent_tokenize
import warnings
import string
from wordcloud import WordCloud
import gensim
import jieba
import re
from underthesea import word_tokenize, pos_tag, sent_tokenize
import regex
import demoji
from pyvi import ViPosTagger, ViTokenizer
import string
from datetime import datetime
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score, roc_auc_score, f1_score, classification_report


#Project

data = pd.read_csv('data_Foody.csv', encoding = 'utf-8', index_col = 0)

##### EDA & Cleaning #####

buffer = io.StringIO()

profile = ProfileReport(data, title="Pandas Profiling Report")

# 10 restaurant có rating cao nhất
rating_hi = data.groupby('restaurant') \
                    .agg({'restaurant':'count', 'review_score':'mean'}) \
                    .rename(columns={'restaurant':'count_restaurant', 'review_score':'mean_review_score'}) \
                    .reset_index() \
                    .sort_values(by='mean_review_score', ascending=False)
                                        
# 10 restaurant số lượt rating nhiều nhất:
review_times = data.groupby('restaurant') \
                    .agg({'restaurant':'count', 'review_score':'mean'}) \
                    .rename(columns={'restaurant':'count_restaurant', 'review_score':'mean_review_score'}) \
                    .reset_index() \
                    .sort_values(by='count_restaurant', ascending=False)

# Creating review_score_level column with level from 0-10
data.loc[ (data['review_score'] >= 0) & (data['review_score'] <= 1.4), 'review_score_level'] = 1
data.loc[ (data['review_score'] > 1.4) & (data['review_score'] <= 2.4), 'review_score_level'] = 2
data.loc[ (data['review_score'] > 2.4) & (data['review_score'] <= 3.4), 'review_score_level'] = 3
data.loc[ (data['review_score'] > 3.4) & (data['review_score'] <= 4.4), 'review_score_level'] = 4
data.loc[ (data['review_score'] > 4.4) & (data['review_score'] <= 5.4), 'review_score_level'] = 5
data.loc[ (data['review_score'] > 5.4) & (data['review_score'] <= 6.4), 'review_score_level'] = 6
data.loc[ (data['review_score'] > 6.4) & (data['review_score'] <= 7.4), 'review_score_level'] = 7
data.loc[ (data['review_score'] > 7.4) & (data['review_score'] <= 8.4), 'review_score_level'] = 8
data.loc[ (data['review_score'] > 8.4) & (data['review_score'] <= 9.4), 'review_score_level'] = 9
data.loc[ (data['review_score'] > 9.4) & (data['review_score'] <= 10), 'review_score_level'] = 10

review_scoreItem_level = data.groupby('review_score_level') \
                            .agg({'review_score_level':'count', 'review_score':'mean'}) \
                            .rename(columns={'review_score_level':'count_review_score_level', 'review_score':'mean_review_score'}) \
                            .reset_index() \
                            .sort_values(by='count_review_score_level', ascending=False)

# Visualization
ax1 = sns.set_style(style=None, rc=None )
fig, ax1 = plt.subplots(figsize=(6,4))
sns.barplot(x='review_score_level', y='count_review_score_level', data=review_scoreItem_level, palette='cubehelix', ax=ax1)
ax2 = ax1.twinx()
fig = sns.lineplot(data=review_scoreItem_level['mean_review_score'], ax=ax2)

# Creating review_score_level column with level with 0 & 1
data.loc[ (data['review_score'] >= 0) & (data['review_score'] <= 6.4), 'review_score_level'] = 0 # Not recommended
data.loc[ data['review_score'] >= 6.5, 'review_score_level'] = 1 # Recommended

#Word Cloud

def print_word_cloud(df_text):
    #Combine all the reviews into one massive string
    review_text = np.array(df_text)
    review_text_combined = " ".join(review for review in review_text)
    # Create stopword list:
    stopwords = set(STOPWORDS)
    #For now let's only remove the
    # stopwords.update(["the"])

    # Generate a word cloud image
    wordcloud = WordCloud(stopwords=stopwords, 
                            background_color="white", 
                            width= 2000, height = 1000, 
                            max_words=50).generate(review_text_combined)

    # Display the generated image:

    image = wordcloud.to_image()
    return image
    
data_0 = data.loc[data['review_score_level'] == 0]
data_1 = data.loc[data['review_score_level'] == 1]

c1 = print_word_cloud(data_1['review_text'])
c0 = print_word_cloud(data_0['review_text'])

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

df = data

df_new = pd.read_csv('df_final.zip', encoding = 'utf-8', index_col = 0)

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#NLP

# Chúng ta sẽ chuẩn bị các stopwords tiếng Việt, emoji, teencode, từ sai, tiếng Anh để xử lý các văn bản hiệu quả.

#VietNamese Stop Words

STOP_WORD_FILE = 'vietnamese-stopwords.txt'

with open(STOP_WORD_FILE,'r',encoding='utf-8') as file:
  stop_words=file.read()

stop_words = stop_words.split('\n')

#EMOJI

EMOJI_CON_FILE = 'emojicon.txt'

with open(EMOJI_CON_FILE,'r',encoding='utf-8') as file:
  emoji=file.read()

emoji = emoji.split('\n')

emoji_dict = {}

for line in emoji:
    key, value = line.split('\t')
    emoji_dict[key] = str(value)
    
#TEENCODE

TEEN_CODE_FILE = 'teencode.txt'

with open(TEEN_CODE_FILE,'r',encoding='utf-8') as file:
  teencode=file.read()

teencode = teencode.split('\n')

teencode_dict = {}

for line in teencode:
    key, value = line.split('\t')
    teencode_dict[key] = str(value)

#WRONG WORD

WRONG_WORDS_FILE = 'wrong-word.txt'

with open(WRONG_WORDS_FILE,'r',encoding='utf-8') as file:
  wrongwords=file.read()

wrongwords = wrongwords.split('\n')

#ENG TO VN

EV_FILE = 'english-vnmese.txt'

with open(EV_FILE,'r',encoding='utf-8') as file:
  e2v=file.read()

e2v = e2v.split('\n')

e2v_dict = {}

for line in e2v:
    key, value = line.split('\t')
    e2v_dict[key] = str(value)

def process_text(text, emoji_dict, teencode_dict, e2v_dict, wrongwords):
    document = text.lower()
    document = document.replace("'","")
    document = regex.sub(r'\.+','.',document)
    new_sentence = ''
    for sentence in sent_tokenize(document):
        # if not(sentence.isascii()):
        #EMOJI
        sentence = ''.join(emoji_dict[word]+' ' if word in emoji_dict else word for word in list(sentence))
        #TEENCODE
        sentence = ' '.join(teencode_dict[word] if word in teencode_dict else word for word in sentence.split())
        #ENGLISH
        sentence = ' '.join(e2v_dict[word] if word in e2v_dict else word for word in sentence.split())
        #Wrong words
        sentence = ' '.join('' if word in wrongwords else word for word in sentence.split())
        new_sentence = new_sentence + sentence + '. '
    document = new_sentence
    #print(doc)
    #DELETE exceed blank space
    document = regex.sub(r'\s+',' ',document).strip()
    return document
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

cr = classification_report(df_new.review_score_level, df_new.preds)

# Save pickle
pkl_filename = 'foody_model.pkl'

with open(pkl_filename, 'rb') as file:  
    lr_model = pickle.load(file)

pkl_count = "count_foody_model.pkl" 

with open(pkl_count, 'rb') as file:  
    count_model = pickle.load(file)



#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#GUI
#Title
st.title("Data Science Project")
st.write("## Foody Sentiment Analysis")

menu = ["Business Objective","EDA & Cleaning","NLP & Machine learning", "New Prediction"]
choice = st.sidebar.selectbox('Menu', menu)

if choice == 'Business Objective':
    st.subheader('Business Objective')
    st.title("I. Tổng quan về Sentiment Analysis")
    st.image("sentiment_1.png")
    st.write("""
    Sentiment analysis (Phân tích cảm xúc) là công nghệ được sử dụng để đo lường xúc cảm trong thông điệp truyền tải dựa vào những đặc điểm được lập trình sẵn dựa trên thang điểm mặc định trong hệ thống, có sự tác động của ngữ cảnh, không gian, thời gian,..
    """)  
    st.write(""" Quá trình này có thể thực hiện bằng việc sử dụng các tập luật (rule-based), sử dụng Machine Learning hoặc phương pháp Hybrid (kết hợp hai phương pháp trên).
    """)
    st.write("""Sentiment Analysis được ứng dụng nhiều trong thực tế, đặc biệt là trong hoạt động quảng bá kinh doanh. Việc phân tích đánh giá của người dùng về một sản phẩm xem họ đánh giá tiêu cực, tích cực hoặc đánh giá các hạn chế của sản phẩm sẽ giúp công ty nâng cao chất lượng sản phẩm và tăng cường hình ảnh của công ty, củng cố sự hài lòng của khách hàng.""")
    st.image("Sentiment-Analysis-La-Gi.jpeg")

    st.title("II. Bussiness Problem")
    st.write("""
    Được xây dựng từ giữa năm 2012 tại TP. HCM, Việt Nam, Foody là cộng đồng tin cậy cho mọi người có thể tìm kiếm, đánh giá, bình luận các địa điểm ăn uống: nhà hàng, quán ăn, cafe, bar, karaoke, tiệm bánh, khu du lịch... tại Việt Nam - từ website hoặc ứng dụng di động. Dù phát triển tốt với ứng dụng di động nhưng Website của Foody vẫn tạo được sức hút với đông đảo người dùng nhờ việc phân chia các mảng riêng biệt dễ quan sát như: Đánh giá, khám phá, đặt bàn (TableNow) và giao hàng (DeliveryNow). Bên cạnh đó, Foody còn cung cấp thêm phần mềm quản lý nhà hàng (FoodyPOS) vô cùng tiện dụng.
    """)
    st.write("""
    Tính đến nay, Foody đã có hàng trăm địa điểm, bình luận và hình ảnh tại các quán ăn từ bình dân đến cao cấp. Có thể nói tra cứu thông tin đánh giá khách quan trên Foody là phương pháp dễ nhất để bạn tiết kiệm thời gian tìm kiếm và lựa chọn ra những địa điểm ăn uống hấp dẫn nhất để trải nghiệm cùng người thân và bạn bè.
    """)
    st.write("""Hệ thống của Foody phân loại các danh mục và địa điểm rất chi tiết. Điều này giúp cộng đồng người dùng của Foody nhanh chóng lọc ra địa điểm đáp ứng tốt mục đích và nhu cầu thiết thực mà không cần tốn kém chi phí, không nhầm lẫn thông tin, không cần mất thời gian chờ đợi như khi tra cứu thiếu định hướng trên internet. Đánh giá từ "khách hàng cũ" trở nên quan trọng hơn bao giờ hết, bởi đây chính là nguồn thông tin tham khảo hữu ích, thiết thực đối với các khách hàng đến sau.
    """)
    st.image("foody_overview.png")
    st.write("""
    Do đó, các nhà hàng/quán ăn cần nỗ lực để cải thiện chất lượng của món ăn cũng như thái độ phục vụ nhằm duy trì uy tín của nhà hàng cũng như tìm kiếm thêm khách hàng mới.
    """)
    st.write(""" Vậy bài toán đưa ra là: Chúng ta cần **xây dựng hệ thống** hỗ trợ nhà hàng/quán ăn phân loại các phản hồi của khách hàng thành các nhóm: tích cực, tiêu cực dựa trên dữ liệu dạng văn bản, nhằm tìm ra các điểm khiến khách hàng chưa hài lòng và đưa ra giải pháp cải thiện thích hợp. 
    """)
    st.image("Using-Sentiment.jpeg")

    st.title("III. How to do")
    st.write("""Các bước cần thực hiện như sau""")
    st.write("""- Thu thập data (càng nhiều thì dự đoán càng chính xác)""")
    st.write("""- Tiền xử lý data""")
    st.write("""- Xây dựng model phù hợp""")
    st.write("""- Dự đoán kết quả""")
    st.image("step.jpeg")



elif choice == "EDA & Cleaning" :
    st.title("EDA & Cleaning")
    #EDA & Cleaning
    st.write('Data Overview') # Dataframe
    st.dataframe(data.head(10))
    #st.write('Pandas Profiling') # Pandas profiling
    #st_profile_report(profile)
    st.write('Overview')
    st.image("diem-review.png")
    st.write("""##### Dễ dàng nhận thấy điểm số từ 8 trở lên chiếm số lượng khá lớn, dựa trên số điểm có thể phỏng đoán các đánh giá thực sự chê hoặc nhận xét không tốt về dịch vụ chiếm số lượng nhỏ hơn các đánh giá tích cực khá nhiều.
    """)
    st.write('10 restaurant có rating cao nhất')
    st.table(rating_hi.head(10))
    st.write('10 restaurant số lượt review nhiều nhất')
    st.table(review_times.head(10))
    st.write('Nhận xét')
    st.markdown(""" <p>- Hầu hết điểm review trung bình từ các bài đều mức khá (7 điểm) trở lên.</p>
                <p>- Các nhà hàng có điểm số cao nhất chưa chắc hẳn sẽ có số lượt review cao nhất, nhưng các nhà hàng có lượng review cao lại đa phần nằm tại khoảng mean. Do đó ta sẽ chú trọng vào Popularity hơn là Rating.</p>""", unsafe_allow_html=True)
    st.code('Chuyển đổi các điểm số từ số thực thành số nguyên')
    ax1 = sns.set_style(style=None, rc=None )
    fig, ax1 = plt.subplots(figsize=(6,4))
    sns.barplot(x='review_score_level', y='count_review_score_level', data=review_scoreItem_level, palette='cubehelix', ax=ax1)
    ax2 = ax1.twinx()
    sns.lineplot(data=review_scoreItem_level['mean_review_score'], ax=ax2)
    st.pyplot(fig=fig, showPyplotGlobalUse = False)
    st.write('Nhận xét')
    st.markdown("""<p> Sau khi chuyển đổi 1 lần chúng ta sẽ dễ dàng thấy rằng các nhà hàng phân hóa thành 2 loại rõ rệt:</p>
                <p>- Điểm >= 8 : Possitive</p>
                <p>- Điểm < 8: Negative</p>""", unsafe_allow_html=True)

    st.write('Recommended Group')               
    st.image("tot.png")
    st.write('Not Recommended Group')
    st.image("xau.png")
    st.write('Nhận xét')
    st.markdown(""" <p> Ta thấy các từ thường gặp trong mỗi class:</p>
                    <p> - Negative: thất_vọng, lỗi, notpositive, tệ, ...</p>
                    <p> - Positive: ok, tốt, nhanh, cẩn_thận, đẹp ...</p>
                    <p> Sau khi có kết quả dự đoán ta sẽ đưa ra các đề nghị thích hợp cho các chủ doanh nghiệp.</p>""", unsafe_allow_html=True)
elif choice == "NLP & Machine learning" :    
    st.title("NLP & Machine learning")
    st.write('Data Pre-processing')
    st.dataframe(df.head(10))
    st.write('Use Model Logistic Regression')
    st.code(cr)
    st.write('Modeling & Evaluation')
    st.dataframe(df_new.head(10))
    st.write('Precision and Recall for each Threshold')  
    st.image("thres.png")
    st.write('ROC curve of class 0')               
    st.image("roc0.png")
    st.write('ROC curve of class 1')  
    st.image("roc1.png")
    st.write('Nhận xét')
    st.markdown(""" <p> Kết quả dự đoán mô hình trên sẽ phần nào giúp các nhà hàng/ quán ăn hiểu được tình hình kinh doanh hiện tại của mình, cũng như hiểu thêm về khách hàng rõ hơn, biết họ đánh giá quán mình như thế nào để cải thiện tốt hơn trong dịch vụ/sản phẩm. </p>""", unsafe_allow_html=True)


elif choice == "New Prediction":
    st.title("New Prediction")
    st.subheader('Select data')
    flag = False
    lines = None
    type = st.radio("Upload data or Input data?", options = ('Upload', 'Input'))
    if type == "Upload":
        #Upload file
        uploaded_file_1 = st.file_uploader("Choose a file", type = ['txt','csv'])
        if uploaded_file_1 is not None:
            lines = pd.read_csv(uploaded_file_1, header=None)
            st.dataframe(lines)
            # st.write(line.columns)
            lines = lines['text']
            flag = True
    if type == "Input":
        review = st.text_area(label="Input your content:")
        if review!="":
            lines = np.array([review])
            # lines = st.dataframe({'text':[review]})
            flag = True
        
    if flag:
        st.write("Content:")
        if len(lines)>0:
            st.code(lines)
            d = {'text':lines}
            lines = pd.DataFrame(data=d)
            lines['text'] = lines['text'].str.lower()
            lines['text'] = lines['text'].str.replace('[\d+]',' ')
            lines['text'] = lines['text'].str.replace('[{}]'.format(string.punctuation), ' ')
            lines['text'] = lines['text'].str.replace("['•','\n','-','≥','±','–','…','_']",' ') 
            lines['text'] = lines['text'].str.replace('(\s[a-z]\s)',' ')
            lines['text'] = lines['text'].str.replace('(\s+)',' ')
            lines['text'] = lines['text'].apply(lambda x: process_text(str(x), emoji_dict, teencode_dict, e2v_dict, wrongwords))
            lines['text'] = lines['text'].apply(lambda x:word_tokenize(x,format = 'text'))
            text_data = np.array(lines['text'])
            bag_of_words = count_model.transform(text_data)
            x_new = bag_of_words.toarray()
            y_pred_new = lr_model.predict(x_new)
            st.write("New predictions (0: Review này mang tính tiêu cực, 1: Review này mang tính tích cực): " + str(y_pred_new))
