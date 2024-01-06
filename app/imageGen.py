from diffusers import DiffusionPipeline, StableDiffusionXLImg2ImgPipeline
import torch
from diffusers.utils import load_image
from PIL import Image
import numpy as np
import requests
import cv2
from . import text_preprocessing



url = ''
class ModelPipeline:
    def __init__(self, url, prompt, negative_prompt, model_path, output_path):
        self.url = url
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.model_path = model_path
        self.output_path = output_path
        self.pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True
        )
        self.pipe.to("cuda")
        self.pipe.load_lora_weights(model_path)

    def input_img(self):
        response = requests.get(url)
        image_array = np.array(bytearray(response.content), dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        init_image = load_image(image).convert("RGB")
        return init_image

    def text_gen(self, input_text):
        prompt = text_preprocessing.Paraphrasing(input_text)
        return prompt

    def run_generate(self):
        generator = torch.Generator("cuda").manual_seed(43) #manual_seed는 랜덤 시드를 고정하는 값
        init_image = self.input_img()
        processed_prompt = self.text_gen(self.prompt)
        image = self.pipe(
            prompt=processed_prompt,
            negative_prompt=self.negative_prompt,
            generator=generator,
            num_inference_steps=20,
            image=init_image,
            output_type="PIL"
        )
        self.save_image(image.images[0])

    def save_image(self, image):
        if isinstance(image, np.ndarray):
            image = (255 * image).astype(np.uint8)
            img = Image.fromarray(image)
        else:
            img = image

        img.save(self.output_path)

# 예시 코드
# 객체 생성 및 메소드 호출
model_pipeline = ModelPipeline(
    url = "/tf/notebook/LEEEUNBEE_1.jpg",
    prompt = "예쁘고 어린 여자, 긴생머리 여자, 아리따운 여자",
    negative_prompt = "noise, mosiac, low quaility",
    model_path="/tf/notebook/3D_Animation_Style-000009.safetensors",
    output_path="/tf/notebook/test.png"
)

model_pipeline.run_generate()
