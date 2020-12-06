# coding=utf-8
"""
-------------------------------------
author : yezhiwen.buaa
introduction : 
-------------------------------------
"""

import os
import shutil

def clean_model(model_dir):
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)