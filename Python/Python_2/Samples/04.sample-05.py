import qrcode

#  site_list.txt는 여러 사이트 주소가 적혀있는 파일
with open('site_list.txt','rt',encoding='UTF8') as f:
    
    read_lines = f.readlines()

    for line in read_lines:
        line = line.strip() # 글자 이외의 것을 없애줌.
        print(line)

        qr_data = line
        qr_image = qrcode.make(qr_data)

        qr_image.save(qr_data + '.png')
