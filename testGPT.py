import myGPT

# q="1번 : 인메모리 데이터 그리드(In-Memory Data Grid)의 역할과 장단점에 대해 설명해주십시오."+"와"+"2번 :인메모리 데이터 그리드(In-Memory Data Grid)가 무엇이며, 어떤 상황에서 사용되는지 설명해주세요."+"1번과 2번이 비슷한 키워드의 질문이면, ture를 완전 다른 키워드의 질문이면 false를 리턴해줘"
# print(myGPT.get_completion(q))
# print(myGPT.first_question(5))
for i in range(1):
    text = myGPT.first_question(4)
    print(text)
    result=text.split("@@@")
    print(result)
    for i in range(2):
        result[i]=result[i].strip()
        result[i]=result[i][2:]
    
    print(result)


