import torch

BATCH_SIZE = 12  # GPU memory size에 맞게
RESIZE_TO = 512 # resize the image training and transforms
NUM_EPOCHS = 100 # number of epochs to train for

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# train image and xml files directory
TRAIN_DIR = "../Microcontroller Detection/train"
# validation image and xml files directory
VALID_DIR = "../Microcontroller Detection/test"
# test image and xml files directory
TEST_DIR = "../Microcontroller Detection/test"

CLASSES = [
    'background', 'Arduino_Nano', 'ESP8266', 'Raspberry_Pi_3', 'Heltec_ESP32_Lora'
]
NUM_CLASSES = 5

# 데이터 로더 생성 후 이미지 시각화 여부
VISUALIZE_TRANSFORMED_IMAGES = False

# location to save model and plots
OUT_DIR = "../outputs"
SAVE_PLOTS_EPOCH = 2 # save loss plots after these many epochs
SAVE_MODEL_EPOCH = 2 # save model after these many epochs

NUM_SAMPLES_TO_VISUALIZE = 10