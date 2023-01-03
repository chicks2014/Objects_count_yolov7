import streamlit as st
import base64
import random as rd
import os
import subprocess
import glob
# from IPython.display import Image, display
# from yolov7 import detect_and_count


st.title('Object counts in Image')
uploaded_files = None

def uploader_callback():
    print('on change called')
    # st.write('on change called')
    if uploaded_files is not None:
        for uploaded_file in uploaded_files:
            my_file = base64.b64encode(uploaded_file.read())
            # print(my_file)

            souce_file = base64.b64decode(my_file)
            file_name = f'source_img_{rd.randint(1, 500)}.jpg'
            print('--' * 50, '\n File uploaded with name: ', file_name, '--' * 50)

            with open(os.path.join('inference\images', file_name), 'wb') as f:
                f.write(souce_file)

            st.image(os.path.join('inference\images', file_name), caption='Sunrise by the mountains')


uploaded_files = st.file_uploader("Choose a file", on_change=uploader_callback, key='file_uploader', accept_multiple_files=True)

if st.button('Generate Inference'):
    st.write('Model is generating Inference...')
    # generate inferences
    result = subprocess.run('python yolov7\detect_and_count.py --weights yolov7\mybest1.pt --conf 0.1 --source inference\images', stdout=subprocess.PIPE, text=True)
    print("\n\n result.stdout -- \n\n")
    print(result.stdout)
    for l in result.stdout.splitlines():
        print(type(l))
        if 'total_counts' in l:
            cnt = l.split('=')[1]
            st.write('Total objects: ', cnt)
        print(l)

if st.button('View Inference'):
    st.write('Inference...')
    i = 0
    limit = 10
    for imageName in glob.glob('runs\detect\exp\*.jpg'):  # assuming JPG
        if i < limit:
            # display(Image(filename=imageName))
            st.image(imageName)
            print("\n")
        i = i + 1
