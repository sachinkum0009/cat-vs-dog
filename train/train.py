"""
script to train the models
"""
import torch

from models.model import CnnModel

def train():
    model = CnnModel()
    model.compile()

