from flask import Flask, render_template, request, redirect, url_for
import os
from PIL import Image  # 이미지 처리를 위한 라이브러리 추가
import imageGen

app = Flask(__name__)

# 웹 페이지 렌더링
@app.route('/')
def index():
    return render_template('index.html')

# 이미지를 업로드하고 처리하는 부분
@app.route('/upload', methods=['POST'])
def upload():
    if request.method == 'POST':
        # 이미지를 업로드하고 저장
        uploaded_file = request.files['image']
        if uploaded_file.filename != '':
            image_path = os.path.join('uploads', uploaded_file.filename)
            uploaded_file.save(image_path)
            
            model_pipeline = imageGen.ModelPipeline(
            url = "",
            prompt = "",
            negative_prompt = "",
            model_path="",
            output_path=""
        )

            image = model_pipeline.run_generate()

            # 새로운 이미지를 로드하여 웹 페이지에 표시
            generated_image = Image.open(image)
            generated_image.show()  # 현재는 이미지를 표시하는 방식으로 표현하고 있습니다.
            
            return redirect(url_for('index'))

# 서버 실행
if __name__ == '__main__':
    app.run(debug=True)
