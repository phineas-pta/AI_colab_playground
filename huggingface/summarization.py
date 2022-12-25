# -*- coding: utf-8 -*-

from transformers import pipeline

model = pipeline(device="cuda:0", task="summarization", model="VietAI/vit5-large-vietnews-summarization")
input_txt = "vietnews: VKS cáo buộc ông Nguyễn Thế Hiệp có sai phạm trong vụ cháy gần Bệnh viện Nhi trung ương khiến 2 người chết, thiệt hại 1,9 tỷ đồng song bị cáo khẳng định vô tội. Mức án đề nghị 9-10 năm tù với bị cáo 73 tuổi được đại diện VKSND quận Ba Đình đưa ra chiều 28/11, quy buộc phạm tội Vi phạm quy định về phòng cháy chữa cháy, theo Điều 313 Bộ luật Hình sự. VKS nhận định ông Hiệp có lỗi trong việc vận hành nhà trọ không phép, không đủ điều kiện an toàn phòng cháy chữa cháy, gây thiệt hại về tài sản và khiến hai người chết. Tuy nhiên, bị cáo chưa bồi thường. Bản luận tội nêu, tại phiên tòa hôm nay ông Hiệp “chưa tỏ thái độ ăn năn hối hận, có nhân thân đặc biệt xấu”. Từ hàng chục năm trước, ông từng 11 lần bị lập danh chỉ bản về hành vi trộm cắp, năm 1985 lại nhận 18 năm tù về các tội cướp tài sản, hiếp dâm, đưa hối lộ …. </s>"
output_txt = model(input_txt)[0]["summary_text"]
print(output_txt)

model = pipeline(device="cuda:0", task="summarization", model="plguillou/t5-base-fr-sum-cnndm")
input_txt = "summarize: Apollo 11 est une mission du programme spatial américain Apollo au cours de laquelle, pour la première fois, des hommes se sont posés sur la Lune, le lundi 21 juillet 1969. L’agence spatiale américaine, la NASA, remplit ainsi l’objectif fixé par le président John F. Kennedy en 1961 de poser un équipage sur la Lune avant la fin de la décennie 1960. Il s’agissait de démontrer la supériorité des États-Unis sur l’Union soviétique qui avait été mise à mal par les succès soviétiques au début de l’ère spatiale dans le contexte de la guerre froide qui oppose alors ces deux pays. Ce défi est lancé alors que la NASA n’a pas encore placé en orbite un seul astronaute. Grâce à une mobilisation de moyens humains et financiers considérables, l’agence spatiale rattrape puis dépasse le programme spatial soviétique."
output_txt = model(input_txt)[0]["summary_text"]
print(output_txt)
