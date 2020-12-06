# coding=utf-8
"""
-------------------------------------
author : yezhiwen.buaa
introduction : 定义feature column
-------------------------------------
"""

class SparseFeat(object):

    def __init__(self, name, vocabulary_size):
        self.name = name
        self.vocabulary_size = vocabulary_size


class DenseFeat(object):

    def __init__(self, name):
        self.name = name
