# 텍스트 데이터 처리 01
import re
text_data = [" Interrobang. By Aishwarya Henriette",
             "Parking and going. By Kear Gua",
             "Today is the noght. By Jar par"] # 텍스트 데이터 생성

# 공백 제거
strip_whitespace = [string.strip() for string in text_data]
print("공백 제거 >>",strip_whitespace)

# 마침표 제거
remove_periods = [string.replace(".", "") for string in strip_whitespace]
print("마침표 제거 >>",remove_periods)
'''
출력 결과
공백 제거 >> ['Interrobang. By Aishwarya Henriette', 'Parking and going. By Kear Gua', 'Today is the noght. By Jar par']
마침표 제거 >> ['Interrobang By Aishwarya Henriette', 'Parking and going By Kear Gua', 'Today is the noght By Jar par']
'''

# 대문자로 바꿔주는 함수
def capitalizer(string: str) -> str: return string.upper()

temp = [capitalizer(string) for string in remove_periods]
print(temp)

# a~z, A~Z를 X로 바꿔주는 함수
def replace_letters_with_X(string:str) -> str:
    return re.sub(r"[a-zA-Z]", "X", string)

data = [replace_letters_with_X(string) for string in remove_periods]
print(data)