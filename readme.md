# Image Generator Project 

### 1. 프로젝트 소개
#### 1.1 개요
- 주제 :  


#### 1.2 목표
- 기존의 이미지에 텍스트의 내용이 반영하여, 새로운 이미지 생성

#### 1.3 기술 스택
- pytorch, AutoPipelineForImage2Image, StableDiffusionPipline, konlpy, numpy, PIL 외



### 2. 프로젝트 프로세스

![image](https://github.com/JinSan-RM/ImageGen_textPlusimage/assets/143769249/5e98129f-483c-4177-a1e4-af4fff2913a3)


이미지 생성 프로세스는 다음과 같은 단계로 실행:

1-1. **텍스트 Input** : 사용자는 이미지를 원하는 형태의 보정 혹은 꾸밈을 위해서 텍스트 입력.
1-2. **이미지 Input** : 사용자는 보정 및 꾸밈을 원하는 이미지를 입력.

2. **Text Prompt RAG 증강 검색** : 입력된 텍스트를 기반으로 RAG 증강 검색을 활용하여 입력된 텍스트를 Augment.

3. **Text preprocessing** : 증강된 텍스트를 텍스트 전처리 단계에 따라서 불필요한 정보를 제거, 필터링, 단어 가중치 적용을 진행

4. **Image Generate Model 활용한 이미지 생성**: 설정된 조건(image, prompt)에 따라 안정적인 확산 모델을 사용하여 이미지를 생성합니다.






----------------------------------------------------------------------
'''
test = ModelPipeline(url, prompt, negative_prompt, model_path, output_path)
'''
