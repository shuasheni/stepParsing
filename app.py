from flask import *

from match.step_graph_match import graph_match
from parse.step_parse import step_parse

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')
@app.route("/stepParse", methods=["POST"])
def upload():
    # 获取file_id参数，假设通过表单数据传递
    file_id = request.form.get('file_id')
    # 获取上传的文件
    step_file = request.files.get('file')

    # 检查参数是否完整
    if not file_id:
        return jsonify({'error': 'Missing file_id parameter'}), 400
    if not step_file:
        return jsonify({'error': 'No file uploaded'}), 400

    fn = f"file\\{file_id}.step"
    step_file.save(fn)

    faces = step_parse(fn)
    feathers = graph_match(fn)

    file_info = {
        'faces': faces,
        'feathers': feathers
    }
    return jsonify(file_info), 200



if __name__ == '__main__':
    app.run()
