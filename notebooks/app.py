import sys, os
import json 
import datetime
import streamlit as st
import copy 
import random
import re
import pickle
import pandas as pd
import time 

if 'filename' not in st.session_state:
    st.session_state['filename'] = None

# Set page config to full width and height
st.set_page_config(
    page_title="SwiftScrub",
    page_icon="🫧",
    layout="wide",
)

# Custom CSS to style the scrollable containers
scrollable_css='''
<style>
    section.main>div {
        padding-bottom: 0rem;
    }
    # [data-testid="stVerticalBlock"]>[data-testid="stHorizontalBlock"]:has([data-testid="stMarkdown"]){
    #     overflow: auto;
    #     max-height: 650px;
    # }
    # [data-testid="element-container"] [data-testid="stTable"]{
    #     overflow: auto; 
    #     max-height: 350px;
    # }
    [data-testid="stExpanderDetails"]:has([data-testid="stTable"]){
        overflow: auto; 
        max-height: 350px;
    }
</style>
'''

if 'file_loaded' not in st.session_state:
    st.session_state['file_loaded'] = False

c1left, c1right= st.container().columns([0.7,0.3])
c1left.markdown("### SwiftScrub🦙&nbsp;🧼&nbsp;:sunglasses:&nbsp;🫧&nbsp;🪪")
h1 = c1right.container().expander(label="Load File", expanded=True)
uploaded_file = h1.file_uploader("File Dialog", on_change=handlefilechange, label_visibility="collapsed")
progress_bar = h1.progress(value=0, text="")
console_container = c1right.container()
console = console_container.status(label="Please load file 📃", state="error")

c3 = c1left.container()
col3 = c1right