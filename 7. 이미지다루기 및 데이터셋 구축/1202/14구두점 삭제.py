# 구두점 삭제
# 구두점 글자의 딕셔너리를 만들어 translate() 적용
import unicodedata
import sys

# 모든 유니코드 구두점을 키로 하고 값은 None인 punctuation 딕셔너리를 만듭니다.
# 문자열로부터 punctuation에 있는 모든 문자를 None으로 바꿉니다. (구두점을 삭제하는 효과)

text_data = ["HI!!!!! I. Love. This. Song.....!!!",
             "10000% Agree?? #AA",
             "Right!!@@"] #구두점이 포함된 텍스트 데이터 생성

# 딕셔너리 형태로 바꾸기( 구두점 문자로 이루어진 딕셔너리를 만듭니다. )
punctuation = dict.fromkeys(i for i in range(sys.maxunicode)
                            if unicodedata.category(chr(i)).startswith('P'))
#print(punctuation)

# 문자열의 구두점 삭제
test = [string.translate(punctuation) for string in text_data]
print(test)

'''
결과 : ['HI I Love This Song', '10000 Agree AA', 'Right']
'''