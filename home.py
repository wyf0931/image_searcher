from flask import Flask
from flask import request, render_template
import descriptor
import searcher
# import json
import cv2
import numpy as np

app = Flask(__name__)

color_index_path = "C:\\Users\\wangyunfei\\Pictures\\color_index.csv"
structure_index_path = "C:\\Users\\wangyunfei\\Pictures\\structure_index.csv"

color_descriptor = descriptor.ColorDescriptor()
structure_descriptor = descriptor.StructureDescriptor()
image_searcher = searcher.Searcher(color_index_path, structure_index_path)


@app.route('/')
def home():
    return app.send_static_file('home.html')


@app.route('/search', methods=['POST'])
def search():
    data = request.files['file'].read()
    image = np.asarray(bytearray(data), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    color_features = color_descriptor.describe(image)
    structure_features = structure_descriptor.describe(image)
    results = image_searcher.search(color_features, structure_features, 10)
    # results = sorted(
    #     results.items(), key=lambda item: item[1], reverse=False)
    # results = results[:10]

    # res = ""
    # for r in searchResults:
    #     tmp = '{\"key\":\"' + str(os.path.basename(
    #         r[0])) + '\", \"score:\":' + str(r[1]) + '},'
    #     print(tmp)
    #     res += tmp
    # return "[" + res[:-1] + "]"

    temp_set = []
    for res in results:
        temp = []
        temp.append("http://127.0.0.1:8080/" + res[0])
        temp.append(res[1])
        temp_set.append(temp)
    return render_template("list.html", results=temp_set)


@app.route('/index', methods=['GET', 'POST'])
def put_index():
    return 'put_index!'


if __name__ == '__main__':
    print("http server run success.")
    app.run()