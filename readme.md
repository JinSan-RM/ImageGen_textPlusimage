# Image Generator Project 


## 🖥 Overview 
입력 받은 프롬포트와 

## 🛠 Requirement 
- Python 3.11.5
- tensorflow 2.15
- scipy 1.11.1
- matplotlib 3.7.2
- pandas 2.0.3
- numpy 1.24.3
- torch 2.1.1
- torchvision 0.16.1
- opencv-python-headless
- fastapi 0.104.1
- konlpy 0.6.0


## ⚙ Project Process
![title](https://github.com/JinSan-RM/ImageGen_textPlusimage/assets/143769249/90709370-25f1-45dd-b4fa-3fd62039c4a2)


### Preprocessing for Prompt
프롬포트는 한국어와 영어 두가지 버전으로 개발되었다.
#### 1️⃣ 한국어 프롬포트
  __Step1. 입력 받은 텍스트와 유사한 문장 다수 생성 - Paraphrasing API__
  
  __Step2. 모든 문장들 토큰화__
  
  __Step3. 조사 제거, 어간 추출__
  
  __Step4. 단어별 가중치 부여__


  
#### 2️⃣ 영어 프롬포트
  __Step1. 입력받은 텍스트와 유사한 문장 다수 생성 - Paraphrasing API__
  
  __Step2. 모든 문장들 영어로 번역 - Papago API__

  __Step3. 모든 영어 문장들 토큰화__
  
  __Step4. stopwords 말뭉치를 활용한 불용어 제거__
  
  __Step5. 단어별 가중치 부여__  *추가 예정*


#### 📌 사용한 API
__Paraphrasing API__


  

__Papago API__

  네이버 개발자 센터에서 제공하는 Open API

  Papago의 인공 신경망 기반 기계 번역 기술(NMT, Neural Machine Translation)로 텍스트를 원하는 언어로 번역하여 반환해준다. 
  * 원본 언어: Korean
  * 목적 언어: English
