from flask import Flask, redirect, url_for, render_template, request, send_file, Response, session, flash
import argparse
import io
from PIL import Image
import datetime
import torch
import cv2
import numpy as np
import tensorflow as tf
from re import DEBUG, sub
from werkzeug.utils import secure_filename, send_from_directory
import os
import subprocess
from subprocess import Popen
import re
import requests
import shutil
import time
import glob
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from ultralytics import YOLO
from os import path
from pfa_flask import create_app
from pfa_flask.views import *
from pfa_flask.views import chatbot
from flask import Flask


app = create_app()


# app.register_blueprint(chatbot, url_prefix='/chatbot')


if __name__=="__main__":
    app.run(debug=True)