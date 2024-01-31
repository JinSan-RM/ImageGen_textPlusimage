from diffusers import DiffusionPipeline, StableDiffusionXLImg2ImgPipeline
from diffusers import DDIMScheduler, DDPMScheduler
import torch
from diffusers.utils import load_image
from PIL import Image
import numpy as np
import requests
import cv2
import bert_prompt
from fastapi import FastAPI
import requests
# import opencv as cv2

app = FastAPI()

class ModelPipeline:
    def __init__(self, prompt):
        # self.url = url    # Image-to-Image 할 때 사용하는 input image 경로
        self.prompt = prompt
        # self.model_path = model_path  # lora 불러다 쓸 경우
        # self.output_path = output_path
        self.pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True
        )
        self.pipe.to("cuda")
        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        # self.pipe.load_lora_weights(model_path)

    def input_img(self):
        response = requests.get(url)
        image_array = np.array(bytearray(response.content), dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        init_image = load_image(image).convert("RGB")
        return init_image

    def text_gen(self, input_text):
        model = bert_prompt.PromptEnglish(input_text)
        agg_texts = model.text_paraphrasing()   # 증강된 한국 문장들
        trans_texts, base_prompt = model.papago_translation(agg_texts)   # 영어로 번역된 문장들
        df = model.bert(trans_texts)    # 키워드, 가중치가 있는 데이터프레임
        positive_prompt = model.positive_prompt(df)
        
        return base_prompt, positive_prompt

    def run_generate(self):
        generator = torch.Generator("cuda").manual_seed(43) # manual_seed는 랜덤 시드를 고정하는 값
        # init_image = self.input_img()

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

    def save_image(self, image):
        
        if isinstance(image, np.ndarray):
            image = (255 * image).astype(np.uint8)
            img = Image.fromarray(image)
        else:
            img = image

        return img.save(self.output_path)

# 예시 코드
# Image,Text-to-Image
# 객체 생성 및 메소드 호출
# model_pipeline = ModelPipeline(
#     url = "/tf/notebook/LEEEUNBEE_1.jpg",
#     prompt = "예쁘고 어린 여자, 긴생머리 여자, 아리따운 여자",
#     negative_prompt = "noise, mosiac, low quaility",
#     model_path="/tf/notebook/3D_Animation_Style-000009.safetensors",
#     output_path="/tf/notebook/test.png"
# )

# model_pipeline.run_generate()
@app.get("/run_generator")
def run_generate(prompt: str):
    # output_path = "C:/Users/weven/Desktop/image.jpg"  # 이미지를 저장할 경로 지정
    model_pipeline = ModelPipeline(prompt)
    model_pipeline.run_generate()
    return "완료"

@app.get("/")
def text(text : str):
    print(text)
    return text
url = ''
