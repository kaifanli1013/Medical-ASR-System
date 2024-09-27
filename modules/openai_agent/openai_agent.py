import os
import json
import pandas as pd
from csv import writer
from openai import OpenAI
from scipy import spatial
from tenacity import retry, wait_random_exponential, stop_after_attempt
from tqdm import tqdm

GPT_MODEL = "gpt-4o"

client = OpenAI()

