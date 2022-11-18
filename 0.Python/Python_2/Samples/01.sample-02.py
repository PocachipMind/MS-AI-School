# 내 컴퓨터의 ip address 를 확인하는 코드.

import socket

print(socket.gethostname()) # 호스트 네임 확인.

in_addr = socket.gethostbyname(socket.gethostname())

print(in_addr)

# 인터넷 프로토콜 (OSI 7계층) 
# 
