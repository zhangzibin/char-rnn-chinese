#!/usr/bin/python
#encoding=utf-8
import sys
reload(sys)
sys.setdefaultencoding('utf8')

from flask import Flask
from flask import jsonify,render_template,request,abort
import redis
import time
import json
import hashlib

app = Flask(__name__)
channel_name = 'cv_channel'

@app.route('/')
def index():
    return render_template('main.html')

@app.route('/api', methods=['POST'])
def api():
    if not request.json or not 'primetext' in request.json:
        abort(400)
    req = {}
    req['text'] = request.json['primetext']
    req['temp'] = request.json['temperature']
    req['seed'] = request.json['seed']
    m = hashlib.md5()
    m.update(str(time.time()))
    req['sid'] = m.hexdigest()

    r = redis.StrictRedis(host='localhost', port=6379, db=0)
    res = r.publish(channel_name, json.dumps(req))
    print res
    if res == 0:
        req['sid'] = 0

    return jsonify({'sid': req['sid']}), 200

@app.route('/res', methods=['POST'])
def res():
    r = redis.StrictRedis(host='localhost', port=6379, db=0)
    sid = request.json['sid']
    responds = r.get(sid)
    if responds is None:
        responds = '0'
    return jsonify({'responds': responds}), 200

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=True)
