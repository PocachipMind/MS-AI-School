import cv2

cap = cv2.VideoCapture(0)
cnt = 0
while (True):
    # Capture Image
    ret, frame = cap.read()
    cv2.imshow('frame', frame)
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # print(frame)

    # f키로 촬영하세요.
    if cv2.waitKey(1) & 0xFF == ord('f'):
        cnt += 1
        print(cnt)
                    # 저장할 폴더 만들어야합니다. 귀찮아서 os dir
        cv2.imwrite(f"./data/temp/{cnt}.png", cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # save image
    # q 연속으로 누르시면 종료됩니다.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    continue

cap.release()
cv2.destroyAllWindows()