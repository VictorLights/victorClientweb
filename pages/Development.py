import streamlit as st 
import os

st.set_page_config(page_title="Systems dev",layout='wide', page_icon='./images/object.png')
st.title('In Development Documentation')
st.title('')

# sidebar content
sidebar = st.sidebar
sidebar.header('Sidebar TOC')

sidebar.header('1.0 Technologies')
sidebar.write('1.1 Artificial Intelligence')
sidebar.header('2.0 Frameworks')
sidebar.write('2.1 OpenCV')
sidebar.header('3.0 Libraries')
sidebar.write('3.1 Python')
sidebar.header('4.0 Tools')
sidebar.write('4.1 Jupyter Lab')
sidebar.header('5.0 Models')
sidebar.write('5.1 Onnx')
sidebar.header('6.0 Concepts')
sidebar.write('6.1 Deep Learning')


st.title('Members Login')
# create an empty container
placeholder = st.empty()

# no database yet, hardcoded credentials
actual_email = "email"
actual_password = "password"

# insert form to the container
with placeholder.form("login"):
    st.markdown("#### Enter your credentials")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    submit = st.form_submit_button("Login")

if submit and email == actual_email and password == actual_password:
    # clear the container and display a success message
    placeholder.empty()
    st.success("Login successful")
elif submit and email != actual_email and password != actual_password:
    st.error("Login failed")
else:
    pass


st.title('Account Registration')
# creating a form
my_form=st.form(key='form-1')
# creating input fields
fname=my_form.text_input('First Name:')
lname=my_form.text_input('Last Name:')
email=my_form.text_input('Email:')

# creating radio button 
gender=my_form.radio('Gender',('Male','Female'))
# creating slider 
age = my_form.slider('Age:',min_value=1,
                         max_value=100,
                         step=1,value=18)
# creating date picker
bday=my_form.date_input('Enter Birthdate:')
# creating a text area
address=my_form.text_area('Account bio:')


# checkbox (Toggle Button)
checkbox = my_form.checkbox("Do you agree to FPT Terms and Conditions?") # return bool (True or False)
if checkbox:
    st.write('You checked the box ✅')
    st.write('Registered Sucessfully')    
    button = st.button('View JSON') # return true or False
    data = {
        "name": fname + " " + lname,
        "email": email,
        "gender": gender,
        "age": age,
        "birthday": str(bday),
        "address": address
    }
    st.json(data)

else:
    st.write('You are currently Unchecked ❌')
    st.write('Registrations Failed')

# creating a submit button
submit=my_form.form_submit_button('Submit')
st.write('')
st.write('')
st.write('')
    

# display media - images and video
st.subheader("Result Output Charts etc")
st.markdown("""
            Model Architecture Documentation future 
            """)
st.image('./media/Resultss.png',
         caption='Custom model uploaded',
         width=800)
st.image('./media/resultsmultigraphs.png',
         caption='Results [ Halong Beer Model ]',
         width=800)
st.image('./media/confusion_matrix.png',
         caption='Confusion Matrix [ Halong Beer Model ]',
         width=800)
st.write('')
st.write('')

st.subheader('Client Tutorial Demo')
video_file = open('./media/projectshowcase.mov',mode='rb').read()
st.video(video_file)
st.write()
st.write()