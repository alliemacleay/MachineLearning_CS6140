__author__ = 'Allison MacLeay'

class Ecoc(object):
    def __init__(self):
        self.functions = []
        self.output_code = []  # one per class - matrix

class EcocFunction(object):
    def __init__(self):
        self.y = []  # one per class
        self.features = []

