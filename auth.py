from flask import Blueprint
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
from .models import User , Cheque
from werkzeug.security import generate_password_hash, check_password_hash
from . import db
from flask_login import login_user, login_required,logout_user, current_user
from flask_login import login_user, logout_user

auth =Blueprint('auth',__name__)

@auth.route("/login", methods=["POST","GET"])
def login():
    if request.method == "POST":
        email = request.form.get('email')
        pwd = request.form.get('pwd')

        user =User.query.filter_by(email=email).first()
        if user:
            if check_password_hash(user.pwd,pwd):
                flash('Logged in successfully !', category='success')
                login_user(user,remember=True)
                return redirect(url_for('views.home'))
            else:
                flash('Incorrect Password , Try Again',category='error')
        else:
            flash('Incorrect Password , Try Again',category='error')

       
    return render_template("login.html",boolean=True)
    

@auth.route("/signup", methods=["POST", "GET"])
def signup():
    if request.method == "POST":
        fullname = request.form.get("fullname")
        email = request.form.get("email1")
        pwd1 = request.form.get("pwd1")
        pwd2 = request.form.get("pwd2")

        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            flash('User with this email already exists. Please log in instead.', category='error')
            return redirect(url_for("auth.login"))

        if len(email) < 8:
            flash('invalid email', category='error')
        elif len(fullname) < 4:
            flash('Name should be at least 5 characters', category='error')
        elif pwd1 != pwd2:
            flash('Passwords dont match', category='error')
        elif len(pwd1) < 7:
            flash('Password should be at least 7 characters', category='error')
        else:
            new_user = User(fullname=fullname, email=email, pwd=generate_password_hash(pwd1, method='pbkdf2:sha256'))
            db.session.add(new_user)
            db.session.commit()
            login_user(new_user, remember=True)
            flash('Account Created Successfully!', category='success')
            return redirect(url_for('auth.login'))

    return render_template("signup.html")



@auth.route("/logout")
def logout():
    logout_user()
    return redirect(url_for("auth.login"))
