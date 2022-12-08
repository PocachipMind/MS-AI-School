import cv2

###### 동영상 속성 확인 ######
cap = cv2.VideoCapture('./video01.mp4') # 0을 넣으면 웹 캡을 가져오게 됩니다. ( 연결 되어있는 카메라 디바이스 가져오는거. )
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
fps = cap.get(cv2.CAP_PROP_FPS)

print('width :', width, 'height :',height )
print('fps :', fps)
print('frame_count: ', frame_count)

'''
width : 1280.0 height : 720.0
fps : 25.0
frame_count:  323.0
'''

###### 동영상 파일 읽기 ######
if cap.isOpened(): # 캡쳐 객체 초기화 확인
    while True:
        ret, frame = cap.read()
        if ret:
            cv2.imshow("video file show", frame)
            cv2.waitKey(25)
        else : # 읽기 실패시 브레이크
            break
else:
    print('비디오 파일 읽기 실패')

cap.release()
cv2.destroyAllWindows()