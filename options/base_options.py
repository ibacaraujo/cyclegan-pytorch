import argparse
import os
from util import util
import torch
import models
import data

class BaseOptions():
    
    def __init__(self):
        self.initialized = self

    def initialize(self):
        pass

    def gather_options(self):
        pass

    def print_options(self):
        pass

    def parse(self):
        pass