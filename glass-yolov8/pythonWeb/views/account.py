from utils import db
from flask import Blueprint, render_template, redirect,request,session,jsonify

#蓝图对象
ac = Blueprint( "account", __name__ )

@ac.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "GET":
        return render_template("login.html")
    
    if request.is_json:
        data = request.get_json()
        name = data.get('username')
        pwd = data.get('password')
    else:
        name=request.form.get("username")
        pwd=request.form.get("pwd")

    user_dict=db.fetch_one("select * from users where name=%s and password=%s", (name, pwd), cache_seconds=60)
    if user_dict :
        session['user_info']={"name":user_dict['name'], "id":user_dict['id']}
        if request.is_json:
            return jsonify({
                "code": 200,
                "message": "登录成功",
                "data": {
                    "token": f"token_{user_dict['id']}_{user_dict['name']}",
                    "name": user_dict['name'],
                    "id": user_dict['id']
                }
            })
        else:
            return redirect("/home")
    
    if request.is_json:
        return jsonify({
            "code": 500,
            "message": "账号或者密码错误"
        })
    else:
        return render_template("login.html", error="账号或者密码错误")
