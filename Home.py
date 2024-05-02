import streamlit as st 

st.set_page_config(page_title="Home",
                   layout='wide',
                   page_icon='./images/home.png')

st.title("DetectXpert: Elavating Visual Insights with AI")
st.caption('"Harnessing the Power of AI Solutions for Retail Merchandising acessible for all." - FPT Software Global Intern')


# info content
st.markdown("""
            ### Large Vision Model (LVM): The next leap forward
            Large models are designed to process massive amounts of data using deep learning techniques. 
            Large language models, such as OpenAI's GPT-3 and Google's BERT, are capable of generating natural language text, 
            answering questions, and even translating between languages. Large visual models, 
            such as OpenAI's CLIP and Google's Vision Transformer, can recognize objects and scenes 
            in images and videos with remarkable precision. By combining these language and visual models, 
            in hopes to create more advanced AI systems that can understand the world in a more human-like way. 
            """)


# tabs content
st.header('Merchandise View')
tab1, tab2, tab3 = st.tabs(['Beer 1','Beer 2','Beer 3'])

with tab1:
    st.write('A taste of home from Halong Bay')
    st.image('./media/halongbeer.jpeg')
with tab2:
    st.write('Saigon the city that never sleeps')
    st.image('./media/saigonbeer.jpeg')
    
with tab3:
    st.write('Tiger Beer all the way from Singapore')
    st.image('./media/tigerbeer.jpeg')


# status elements content
st.success("Free Tier Option A: 50USD/month")
st.markdown("""
- [Click here to begin](/Object_Detection_Image/)  
            """)
st.warning("Free Tier Option B: 100USD/month")
st.markdown("""
- [Click here to begin](/YOLO_for_image/)  
            """)
st.error("Free Tier Option C: 200USD/month")
st.markdown("""
- [Click here to begin](/YOLO_for_image/)  
            """)