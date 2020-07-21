import sys 
import ast
sys.path.insert(0, './jnet_inference')
from jnet_inference.nets import Model,load_model_from_file
from jnet_inference.evaluate import evaluate