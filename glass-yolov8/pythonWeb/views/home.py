from flask import Blueprint, render_template

#蓝图对象
ho = Blueprint( "home", __name__ )

@ho.route("/home")
def home():
    return render_template("home.html")