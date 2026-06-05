from flask import Blueprint, render_template

#蓝图对象
se = Blueprint( "select_fan", __name__ )

@se.route("/select_fan")
def select_fan():
    return render_template("select_fan.html")