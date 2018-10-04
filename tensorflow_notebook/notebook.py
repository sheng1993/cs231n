#%%
import tensorflow as tf
from tensorflow import keras
from models import VGG
print(tf.__version__)

#%%
model = VGG()

#%%
model.get_layer('reshape_1')