import torch.nn as nn
import torch
import torch.nn.functional as F
import pandas as pd
import os
import torchaudio
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch.nn.functional as F

from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor, Wav2Vec2CTCTokenizer, Wav2Vec2Processor, WavLMModel, AutoProcessor
import numpy as np
import librosa
import pickle
import json
import gc
import math
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import argparse
import yaml
