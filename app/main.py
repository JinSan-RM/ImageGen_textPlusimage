from flask import Flask, render_template, request, redirect, url_for, send_file
import os
from PIL import Image  # 이미지 처리를 위한 라이브러리 추가
from diffusers import DiffusionPipeline, StableDiffusionXLImg2ImgPipeline
from diffusers import DDIMScheduler, DDPMScheduler
import torch
from diffusers.utils import load_image
from PIL import Image
import numpy as np
import requests
import cv2
import bert_prompt
import requests
import imageGen

app = Flask(__name__)

class ModelPipeline:
    def __init__(self, prompt, output_path):
        # self.url = url    # Image-to-Image 할 때 사용하는 input image 경로
        self.prompt = prompt
        # self.model_path = model_path  # lora 불러다 쓸 경우
        self.output_path = output_path
        self.pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True
        )
        self.pipe.to("cuda")
        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        # self.pipe.load_lora_weights(model_path)

    # def input_img(self):
    #     response = requests.get(url)
    #     image_array = np.array(bytearray(response.content), dtype=np.uint8)
    #     image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    #     init_image = load_image(image).convert("RGB")
    #     return init_image

    
    # def save_image(self, image):
    #     if isinstance(image, np.ndarray):
    #         image = (255 * image).astype(np.uint8)
    #         img = Image.fromarray(image)
    #     else:
    #         img = image

    #     img.save(self.output_path)
    #     return self.output_path

    # 프롬프트 생성 코드 
    def text_gen(self, input_text):
        model = bert_prompt.PromptEnglish(input_text)
        agg_texts = model.text_paraphrasing()   # 증강된 한국 문장들
        trans_texts, base_prompt = model.papago_translation(agg_texts)   # 영어로 번역된 문장들
        df = model.bert(trans_texts)    # 키워드, 가중치가 있는 데이터프레임
        positive_prompt = model.positive_prompt(df)
        return base_prompt, positive_prompt

    def run_generate(self):
        generator = torch.Generator("cuda").manual_seed(43) # manual_seed는 랜덤 시드를 고정하는 값
        # init_image = self.input_img() #Image-to-Image할 떄 사용 
        
        # 입력값에 따라 생성되는 prompt와 prompt_2
        processed_prompt, key_words = self.text_gen(self.prompt)
        positive_prompt = 'realism++, realistic++, true-to-life++, photo+++, best quality++, balanced++, detailed++, well-defined++, clear++, realistic color palette++, soft-focused realism++'

        image = self.pipe(
            prompt = processed_prompt,
            prompt_2 = key_words + positive_prompt,
            negative_prompt = 'animation, anime, doll, disney, cartoons, cropped, misshapen, blurry, unfocused, desaturated, abstract, surreal, pixelated, noisy, pop art, no faces, no objects, no landscape, mutilated, disfigured, ugly, deformed, no blurry faces, clear visibility of faces, no abstract faces, avoid shadows on faces',
            generator=generator,
            num_inference_steps=20,
            guidance_scale=7.5,
            # image=init_image,
            output_type="PIL"
        )
        
        return self.save_image(image.images[0])

modelpipe = ModelPipeline('안녕하세요', 'test.png')

# @app.route('/show/<prompt>')
# def show(prompt):
#     model_pipeline = ModelPipeline(
#         prompt = "", 
#         output_path = "temp_image.png"
#     )
    
#     image_path = model_pipeline.run_generate()
    
#     return send_file(image_path, mimetype='image/png', as_attachment=True)


# # 서버 실행
# if __name__ == '__main__':
#     app.run(debug=True)


# @app.route('/')
# def hello():
#     return '<h2>Hello, joowon<h2>'

# 웹 페이지 렌더링
# @app.route('/')         # methods ='GET'   써야해요?
# def index():
#     return render_template('index.html')

# # 이미지를 업로드하고 처리하는 부분
# @app.route('/upload')
# def upload():
#     if request.method == 'POST':
#         # 이미지를 업로드하고 저장
#         uploaded_file = request.files['image']
#         if uploaded_file.filename != '':
#             image_path = os.path.join('uploads', uploaded_file.filename)
#             uploaded_file.save(image_path)
            
#             model_pipeline = imageGen.ModelPipeline(
#             # url = "",
#             prompt = "",
#             # negative_prompt = "",
#             # model_path="",
#             # output_path=""
#         )

#             image = model_pipeline.run_generate()

#             # 새로운 이미지를 로드하여 웹 페이지에 표시
#             generated_image = Image.open(image)
#             generated_image.show()  # 현재는 이미지를 표시하는 방식으로 표현하고 있습니다.
            
#             return redirect(url_for('index'))