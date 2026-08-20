## GAN 논문리뷰

- introduction
    
    짝을 이루지 않은 이미지 간 변환에는 잘 정의되거나 합의된 평가 지표가 없다.
    
    지금까지 이미지 번역 모델의 성공은 제한된 수의 이미지에 대한 주관적이고 질적인 시각적 비교를 기반으로 했다. (정성적 평가)
    
    별도의 분류기를 사용하여 내용과 스타일을 모두 고려하는 이미지 대 일러스트 모델의 정량적 평가를 위한 새로운 프레임워크를 제안한다.
    
    style transfer 영역에 새로운 “일러스트레이션” 이미지를 적용한다. (미술작품과 만화와는 다른)
    
    <img src="https://github.com/eunbinni/CV/blob/main/GANILLA/compare.png" width="500" height="300"/>
                                                                                            
    CycleGAN/DualGAN에서는 좋지 않은 결과가 나온다.
    
    style과 content의 균형을 잘 맞추는 새로운 네트워크를 제안한다. 
    
    dataset : 363개의 다른 책과 24개의 다른 예술가로부터 온 9448개의 삽화가 포함되어 있다.
    
- Related work
    
    style transfer 에 대한 두가지 접근법
    
    1. CNN을 사용하여 그림이나 일러스트레이션이 될 수 있는 주어진 스타일 이미지로 입력 이미지를 스타일링한다.
    2. GAN 이용
    ① 쌍을 이루는 이미지 - 데이터 수집이 어려움
    ② 쌍을 이루지 않는 이미지 ex) DualGAN, CycleGAN → CartoonGAN은 CycleGAN의 순환 구조를 대체하기 위한 새로운 훈련 프레임워크를 제시하고 새로운 loss function까지 도입했다.
- GANILLA
    
    unpaired data를 가진 model은 fail to transfer style and content at the same time → 이 문제 해결을 위한 framework : GANILLA
    
    GANILLA는 style을 transfer하는 동안 content를 보존하기 위해 low-level features을 활용한다.
    
    ## Generator
    
    ### 1. **downsampling**
    
    ResNet-18 모델을 변형한 것
    
    7 * 7 kernel을 가진 하나의 convolution layer - instance norm - max pooling layer
    
    conv - instance norm - conv - instance norm - concat
    
    ### 2. upsampling
    
    업샘플링(가장 가까운 이웃) 작업 전에 긴 스킵 연결(그림 2의 파란색 화살표)을 통해 다운샘플링 단계에서 각 레이어의 출력을 사용하여 하위 레벨의 피처를 공급한다.
    
    layer IV의 출력은 conv layer와 upsample layer를 통해 공급되어 이전 layer의 feature 크기와 일치하기 위해 feature map 크기를 증가시킨다.
    
    이 단계에서 모든 conv filter는 1 × 1 kernel을 가지고 있다. 
    
    마지막으로, convolution layer with 7 × 7 kernel가 3 channel translated image(output)를 출력하는데 사용된다.
    
    한 layer안에 두개의 블록이 있고, 3개의 conv layer가 있다
    
    layer는 총 4개로 구성
    
    ## Discriminator
    
    70 × 70 PatchGAN
    
    ## Training Options
    
    - loss function
    1. Minimax losses
    2. cycle consistency loss - L1 distance 사용
    - cycle-consistency
    
    첫 번째 세트(G)는 소스 이미지를 대상 도메인에 매핑
    
    두 번째 세트(F)는 입력을 대상 도메인 이미지로 가져가고 주기적인 방식으로 소스 이미지를 생성하려고 한다.
    
    - Dataset
    
    256 * 256 픽셀
    
    **Training dataset : 5402 images**
    
    **Test dataset : 751 images**
    
    source / target, 두가지의 unpaired image
    
    natural images as source domain / illustration images as target domain
    
    (논문에서 source는 content image / target는 style image)
    
    - Learning rate: 0.0002
    - Solver: Adam
    - Epoch: 200
    
- Evaluation
    
    ### **성능 측정을 위한 새로운 framework**
    
    GAN 생성 삽화의 품질을 결정할 수 있는 두가지 요인 : target **스타일을 갖는것, 내용 보존**
    
    1. Style-CNN :  스타일 전송 측면에서 결과가 얼마나 좋은지 측정하는 것을 목표로 한다.
    
    스타일별 분류기를 훈련시키기 위해 스타일을 유지하며 시각적 콘텐츠에서 훈련 이미지 분리한다. 
    
    일러스트레이션 이미지에서 무작위로 자른 작은 패치(100 × 100 픽셀) 사용
    
    이 패치를 사용하여 스타일 분류기인 Style-CNN을 훈련시킨다.
    
    Training set : 삽화 아티스트를 위한 10개 클래스 / 자연 이미지를 위한 1개 클래스 존재
    
    분류기를 테스트하기 위해 **생성된 이미지만** 사용.
    
    1. Content-CNN : 입력 콘텐츠의 보존 여부를 탐지하는 것을 목표로 한다. 
    
    특정 장면 범주 (숲, 거리 등)을 콘텐츠로 정의
    
    Training set : 4150개 / Test set: 500개
    
    특정 스타일로 산 이미지를 생성한다면 생성 이미지 또한 산 이미지로 분류되어야 한다.
    
    생성된 이미지가 콘텐츠와 연결이 끊긴 경우 그림으로 분류될 수 있다.(예: 음수)
    
    내용이 보존되지 않으면 장면의 특정 정의가 없는 일러스트레이션에 불과하기 때문.
    
    ### **모델 성능 비교**
    
    기준 : Train Time, No of Parms
    
    1. GANILLA
    
    고유한 아티스트 스타일로 이미지 생성
    
    약간의 결함 존재한다.
    
    1. CartoonGAN & DualGAN
    
    콘텐츠를 보존하지만 많은 예시에서 스타일을 전달하는 데 부족하다.
    
    DualGAN은 가장 높은 content 점수를 받았다.
    
    1. CycleGAN
    
    일러스트레이터들의 스타일을 잘 포착하지만, 내용이 무엇인지 구분하기 어렵다. 
    
    또한 생성된 이미지에서 소스 일러스트에서 얼굴과 사물과 같은 것들을 환각시킵니다.
    
    Style 면에서 다른 모델에 비해 뛰어나다.
    
    - 절제 실험
        
        ### 목적 : 모델의 효과를 자세히 평가하기 위해
        
        1. downsampling : 다운 샘플링 CNN → 원본 ResNet-18로 교체
        2. upsampling : deconv layer가 있는 downsampling CNN 사용
        
        ### downsampling(Ablation Model 1)
        
        - GANILLA와 유사한 콘텐츠 점수
        - 스타일 점수가 GANILLA보다 낮다.
        - 기존의 ResNet-18 구조를 수정하여 GANILLA가 입력 이미지를 성공적으로 스타일화할 수 있음을 시사한다.
        
        ### upsampling(Ablation Model 2)
        
        GANILLA보다 스타일 점수가 높다.
        
         하지만 내용 점수가 너무 낮다.
        
        업 샘플링 부분에서 낮은 수준의 기능을 사용하는 것이 content를 보존하는 부분에 큰 도움이 됨을 시사한다.
        
- Conclusions
    - illustration 데이터 세트와 image-illustration 번역을 위한 새로운 Generator를 제시하였다.
    - illustration은 매우 추상적인 대상과 모양을 담고 있기 때문에, 현재의 최첨단 발전기 네트워크는 내용과 스타일을 동시에 전달하지 못한다.
    
    → solution : GANILLA은 업샘플링 부분뿐만 아니라 다운샘플링 상태에서도 낮은 수준의 기능을 사용한다.
    
    - Generator 모델을 평가하기 위한 잘 정의된 지표가 없다.
    
    → solution :  content과 style 측면에서 이미지 간 변환 모델을 정량적으로 평가하는 두개의 CNN을 기반으로 하는 새로운 평가 프레임워크를 제안한다.
