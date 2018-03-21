from flask.json import jsonify

def estimation(text,img_num):
    len_text = len(text)
    img_num = int(img_num)
    read_time = (len_text/1000)*60 + img_num*12
    return jsonify({'time': read_time, 'unit': 'sec' })