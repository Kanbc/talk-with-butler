from flask.json import jsonify

def estimation(text,img_num):
    len_text = len(text)
    img_num = int(img_num)
    read_time = float((len_text/1000) + img_num*0.2)
    if read_time < 1:
        return jsonify({'time': int(1), 'unit': 'min' })
    return jsonify({'time': int(round(read_time)), 'unit': 'min' })