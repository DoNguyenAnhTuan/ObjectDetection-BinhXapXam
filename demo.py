# Thư viện học máy, sử dụng để tải model YOLOv5 cho nhận diện đối tượng
import torch

# Thư viện tạo giao diện web cho model học máy, có thể dùng để triển khai code thành ứng dụng web đánh giá bài poker
import gradio as gr

# Thư viện tạo biểu đồ và hình ảnh
import matplotlib.pyplot as plt

# Hỗ trợ tải và hiển thị hình ảnh
import matplotlib.image as mpimg

# Thư viện xử lý ảnh (mở, thao tác hình ảnh)
from PIL import Image, ImageEnhance

# Cung cấp hàm tạo hoán vị và tổ hợp (đánh giá các kiểu bài khác nhau)
from itertools import permutations, combinations

# Thư viện tính toán số
import numpy as np

#  Tạo mã định danh duy nhất
import uuid

# Tạo số ngẫu nhiên
import random

# Tương tác với hệ điều hành (truy cập tệp, thư mục)
import os

# Tải model yolo5s từ kho lưu trữ
model = torch.hub.load('ultralytics/yolov5', 'custom', path='Nhom3-BinhXapXam.pt', force_reload=True) 

# đường dẫn vào thư mục
path_52cards = "52cards/"
# định dạng tiền tố của ảnh
prefix = "800px-Playing_card_"
# định dạng hấu tố của ảnh
suffix = ".svg.png"

suits = {"spade":0, "club":1, "diamond":2, "heart":3}
numbs = {None:0, "2":1, "3":2, "4":3, "5":4, "6":5, "7":6, "8":7,
         "9":8, "10":9, "J":10, "Q":11, "K":12, "A":13}

exp_14 = [1, 14, 196, 2744, 38416, 537824]
ranks = {
    "high_card": 0, # Mậu thầu
    "pair": 1, #đôi
    "two_pairs": 2, # 2 đôi
    "three_of_a_kind": 3, # 3 lá bài cùng giá trị
    "straight": 4, # sảnh không đồng chất
    "flush": 5, # Thùng-đồng chất
    "full_house": 6, #cù lũ - 1 bộ ba và 1 bộ 2
    "four_of_a_kind": 7, # tứ quý
    "straight_flush": 8, # thùng phá sảnh
}

A2345_combo = ("2", "3", "4", "5", "A")
all_of_straights = [("2", "3", "4", "5", "6")]
all_of_straights += [("3", "4", "5", "6", "7")]
all_of_straights += [("4", "5", "6", "7", "8")]
all_of_straights += [("5", "6", "7", "8", "9")]
all_of_straights += [("6", "7", "8", "9", "10")]
all_of_straights += [("7", "8", "9", "10", "J")]
all_of_straights += [("8", "9", "10", "J", "Q")]
all_of_straights += [("9", "10", "J", "Q", "K")]
all_of_straights += [("10", "J", "Q", "K", "A")]

all_of_straights_3ele = [("2", "3", "A")]
all_of_straights_3ele += [("2", "3", "4")]
all_of_straights_3ele += [("3", "4", "5")]
all_of_straights_3ele += [("4", "5", "6")]
all_of_straights_3ele += [("5", "6", "7")]
all_of_straights_3ele += [("6", "7", "8")]
all_of_straights_3ele += [("7", "8", "9")]
all_of_straights_3ele += [("8", "9", "10")]
all_of_straights_3ele += [("9", "10", "J")]
all_of_straights_3ele += [("10", "J", "Q")]
all_of_straights_3ele += [("J", "Q", "K")]
all_of_straights_3ele += [("Q", "K", "A")]

# Chia tách chuỗi bài (ví dụ: "800px-Playing_card_club_2.svg.png") thành bộ bài và giá trị ("800px-Playing","card","club","2.svg.png")
# phân tách lá bài và giá trị cấp bậc
def card_split(suit_card):
    tmp = suit_card.split("_")
    return tmp[0], tmp[1]

# Xác định loại chi dựa trên thứ hạng và chất của 5 lá bài

def identify_combo(combo): # card_1 < card_2 < .. < card_5
    if len(combo) == 3:
        card_1, card_2 = None, None
        card_3, card_4, card_5 = combo
        s1, n1 = None, None
        s2, n2 = None, None
    else:
        card_1, card_2, card_3, card_4, card_5 = combo
        s1, n1 = card_split(card_1)
        s2, n2 = card_split(card_2)
    
    s3, n3 = card_split(card_3)
    s4, n4 = card_split(card_4)
    s5, n5 = card_split(card_5)
    
    # S là suits - Chất của bài
    # N là numbs - Giá trị của bài
    # Straight flush ----------------------------| thùng phá sảnh
    if s1 == s2 == s3 == s4 == s5: #chất
        if (n1, n2, n3, n4, n5) in all_of_straights: #giá trị
            return "straight_flush", n5, None, None, None, None
        if (n1, n2, n3, n4, n5) == A2345_combo:
            return "straight_flush", n4, None, None, None, None
        # Flush ---------------------------------|
        return "flush", n5, n4, n3, n2, n1
    
    # Four of a kind ----------------------------| tứ quý
    if n1 == n4:
        return "four_of_a_kind", n4, None, None, None, None
    if n2 == n5:
        return "four_of_a_kind", n5, None, None, None, None
    
    # Full House --------------------------------| cù lũ 3-2
    if n1 == n3 and n4 == n5:
        return "full_house", n3, None, None, None, None
    if n1 == n2 and n3 == n5:
        return "full_house", n5, None, None, None, None

    # Straight ----------------------------------| sảnh
    if (n1, n2, n3, n4, n5) == A2345_combo:
        return "straight", n4, None, None, None, None # it means that combo is 2-3-4-5-A
    if (n1, n2, n3, n4, n5) in all_of_straights:
        return "straight", n5, None, None, None, None
    
    # Three of a kind ---------------------------| sám cô 3c giống nhau
    if n1 == n3:
        return "three_of_a_kind", n3, None, None, None, None
    if n2 == n4:
        return "three_of_a_kind", n4, None, None, None, None
    if n3 == n5:
        return "three_of_a_kind", n5, None, None, None, None
    
    # Two pairs ---------------------------------| 2 đôi
    if n1 == n2 and n1 != None: 
        if n3 == n4:
            return "two_pairs", n4, n2, n5, None, None
        if n4 == n5:# lý do không có n3 ==n5 vì sắp xếp theo từ bé đến lớn
            return "two_pairs", n5, n2, n3, None, None
    else:
        if n2 == n3 and n4 == n5:
            return "two_pairs", n5, n3, n1, None, None
    
    # Pair --------------------------------------| 1 đôi
    if n1 == n2 and n1 != None:
        return "pair", n2, n5, n4, n3, None
    if n2 == n3:
        return "pair", n3, n5, n4, n1, None
    if n3 == n4:
        return "pair", n4, n5, n2, n1, None
    if n4 == n5:
        return "pair", n5, n3, n2, n1, None
    
    # High card ---------------------------------| mậu thầu
    return "high_card", n5, n4, n3, n2, n1

# So sánh thứ hạng của 2 lá bài
def numbs_comparison(n1, n2):
    if n1 == n2 == None: # giá trị
        return 0
    if n1 == None: #n1 k hợp lệ => n1<n2 trả về -1
        return -1
    if n2 == None: #n2 k hợp lệ => n1>n2 trả về 1
        return 1
    
    if numbs[n1] < numbs[n2]:
        return -1
    if numbs[n1] > numbs[n2]:
        return 1
    return 0
    
# So sánh sức mạnh của 2 chi dựa trên loại chi và thứ hạng lá bài
def comboes_comparison(combo_1, combo_2):
    identify_1, r11, r12, r13, r14, r15 = identify_combo(combo_1)
    identify_2, r21, r22, r23, r24, r25 = identify_combo(combo_2)
    i_in_1 = ranks[identify_1]
    i_in_2 = ranks[identify_2]
    if i_in_1 > i_in_2: #chi 1 mạnh hơn chi 2
        return 1
    if i_in_1 < i_in_2: #chi 2 mạnh hơn chi 1
        return -1
    #nếu 2 chi bằng nhau. So sách từng con bài
    if identify_1 == "straight_flush": # thùng phá sảnh
        print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
        return numbs_comparison(r11, r21)
    
    if identify_1 == "four_of_a_kind": # tứ quý
        return numbs_comparison(r11, r21)
    
    if identify_1 == "full_house": #cù lũ
        return numbs_comparison(r11, r21)
    
    if identify_1 == "flush": # thùng. So sánh từng lá bài đến khi tìm ra lá phân định thắng thua
        comp_1 = numbs_comparison(r11, r21)
        if comp_1 != 0:
            return comp_1
        comp_2 = numbs_comparison(r12, r22)
        if comp_2 != 0:
            return comp_2
        comp_3 = numbs_comparison(r13, r23)
        if comp_3 != 0:
            return comp_3
        comp_4 = numbs_comparison(r14, r24)
        if comp_4 != 0:
            return comp_4
        return numbs_comparison(r15, r25)
    
    if identify_1 == "straight":# sảnh
        return numbs_comparison(r11, r21)
        
    if identify_1 == "three_of_a_kind":# xám cô
        return numbs_comparison(r11, r21)
    
    if identify_1 == "two_pairs":# 2 đôi
        comp_1 = numbs_comparison(r11, r21)
        if comp_1 != 0:
            return comp_1
        comp_2 = numbs_comparison(r12, r22)
        if comp_2 != 0:
            return comp_2
        return numbs_comparison(r13, r23)
    
    if identify_1 == "pair": # đôi
        comp_1 = numbs_comparison(r11, r21)
        if comp_1 != 0:
            return comp_1
        comp_2 = numbs_comparison(r12, r22)
        if comp_2 != 0:
            return comp_2
        comp_3 = numbs_comparison(r13, r23)
        if comp_3 != 0:
            return comp_3
        return numbs_comparison(r14, r24)
    
    # identify_1 == "high card" # mậu thầu. So sánh từng lá bài đến khi tìm ra lá phân định thắng thua
    comp_1 = numbs_comparison(r11, r21)
    if comp_1 != 0:
        return comp_1
    comp_2 = numbs_comparison(r12, r22)
    if comp_2 != 0:
        return comp_2
    comp_3 = numbs_comparison(r13, r23)
    if comp_3 != 0:
        return comp_3
    comp_4 = numbs_comparison(r14, r24)
    if comp_4 != 0:
        return comp_4
    return numbs_comparison(r15, r25)

# So sánh sức mạnh của 3 chi
def hands_comparison(hands_1, hands_2):
    combo_1_1, combo_1_2, combo_1_3 = hands_1
    combo_2_1, combo_2_2, combo_2_3 = hands_2
    value = comboes_comparison(combo_1_1, combo_2_1)
    value += comboes_comparison(combo_1_2, combo_2_2)
    value += comboes_comparison(combo_1_3, combo_2_3)
    return value
# value > 0 chi 1 thắng
# value < 0 chi 2 thắng 
# value = 0  2 chi bằng nhau


# Tính toán điểm cho một chi dựa trên loại chi và thứ hạng lá bài
def scores_computation(combo):
    identify, r1, r2, r3, r4, r5 = identify_combo(combo)
    score = ranks[identify] + numbs[r1]/exp_14[1] + numbs[r2]/exp_14[2]   + numbs[r3]/exp_14[3] + numbs[r4]/exp_14[4] + numbs[r5]/exp_14[5]
    return score
# 1, 14, 196, 2744, 38416, 537824
# numbs = {None:0, "2":1, "3":2, "4":3, "5":4, "6":5, "7":6, "8":7,
        #  "9":8, "10":9, "J":10, "Q":11, "K":12, "A":13}
# score sẽ được tính toán như sau:
# ('spade_3', 'spade_4', 'spade_7', 'spade_8', 'spade_K'), ('club_2', 'diamond_4', 'diamond_7', 'club_10', 'diamond_10'), ('heart_3', 'heart_8', 'heart_J')
# score= 5+12/14+7/196+6/2744+3/38416+2/537824=5.89512554293
# score= 1+10/14+9/196+7/2744+3/38416+2/537824=1.76283691319
# score= 0+10/14+7/196+3/2744=0.75109329446
# ~ 8.3

# Tạo tất cả các tổ hợp 3 chi từ 13 lá bài và tính điểm cho mỗi tổ hợp
def permutations_553(hands):
    
    # Kiểm tra có thắng tuyệt đối không
    hands_suit = dict()
    hands_num = dict()
    # Tạo từ điển đếm chất và số:
    for card in hands:
        suit, num = card_split(card)
        # Nếu chất có thì thôi còn chưa thì + thêm 1
        if suit not in hands_suit.keys():
            hands_suit[suit] = 1
        else:
            hands_suit[suit] += 1
        # Nếu giá trị có thì thôi còn chưa thì + thêm 1
        if num not in hands_num.keys():
            hands_num[num] = 1
        else:
            hands_num[num] += 1
            
    perfect_win = len(hands_num) == 13 # 13 lá khác nhau từ 2 đến A 

    if perfect_win:
        combo_1 = (hands[0], hands[1], hands[2], hands[3], hands[4])
        combo_2 = (hands[5], hands[6], hands[7], hands[8], hands[9])
        combo_3 = (hands[10], hands[11], hands[12])
        hand = [combo_1, combo_2, combo_3]
        return [hand], [float('inf')]
    
    perm_553 = []
    scores_553 = []
    # Tạo ra danh sách từ thưu viện tổ hợp 5 lá
    comboes_1 = list(combinations(hands, 5))
    for c1, c2, c3, c4, c5 in comboes_1:
        combo_1 = (c1, c2, c3, c4, c5)
        # sao chép  danh sách bài sang tệp tạm để không ảnh hưởng bài gốc
        tmp_hands = hands.copy()
        tmp_hands.remove(c1)
        tmp_hands.remove(c2)
        tmp_hands.remove(c3)
        tmp_hands.remove(c4)
        tmp_hands.remove(c5)
        
        comboes_2 = list(combinations(tmp_hands, 5))
        for c6, c7, c8, c9, c10 in comboes_2:
            combo_2 = (c6, c7, c8, c9, c10)

            combo_3 = tmp_hands.copy()
            combo_3.remove(c6)
            combo_3.remove(c7)
            combo_3.remove(c8)
            combo_3.remove(c9)
            combo_3.remove(c10)
            # khai báo combo 3 thuộc kiểu dữ liệu bất biến k thay đổi chắc chắn là 3 lá
            combo_3 = tuple(combo_3)
            
            score_1 = scores_computation(combo_1)
            # print("diem 1",score_1)
            score_2 = scores_computation(combo_2)
            # print("diem 2",score_2)
            score_3 = scores_computation(combo_3)
            # print("diem 3",score_3)

            ident_1 = int(score_1)
            # print("ident_1",ident_1)
            ident_2 = int(score_2)
            # print("ident_2",ident_2)
            ident_3 = int(score_3)
            # print("ident_3",ident_3)

            if ident_1 >= ranks["four_of_a_kind"]: #7
                perm_553.append([combo_1, combo_2, combo_3])
                scores_553.append(float("inf"))
            else:
                if score_1 >= score_2 and score_2 >= score_3:
                    vip_hands = [combo_1, combo_2, combo_3]
                    
                    perm_553.append(vip_hands)
                    if ident_1 == ident_2 == 5:  # two of flushes
                        card_1, card_2, card_3 = combo_3
                        s1, n1 = card_split(card_1)
                        s2, n2 = card_split(card_2)
                        s3, n3 = card_split(card_3)
                        if s1 == s2 == s3:
                            score = float('inf') # Trường hợp 3 thùng
                    elif ident_1 == ident_2 == 4: # two of straights
                        if combo_3 in all_of_straights_3ele:
                            score = float('inf') # trường hợp 3 sảnh
                    elif ident_1 == 6 and ident_2 == 6: # two of full houses
                        s1, n1 = card_split(card_1)
                        s2, n2 = card_split(card_2)
                        s3, n3 = card_split(card_3)
                        if n1 == n3: # trường hợp 6 đôi
                            score = float('inf')
                    else:
                        score = score_1 + score_2 + score_3
                    scores_553.append(score)
    return perm_553, scores_553
  
#  Sắp xếp 5 lá bài theo thứ hạng và chất
def sort_n2(a):
    # S là suits - Chất của bài
    def cards_comparison(card_1, card_2):
        s1, c1 = card_split(card_1)
        s2, c2 = card_split(card_2)
        #thứ hạng lớn hơn thì return 1
        if numbs[c1] > numbs[c2]:
            return 1
        #thứ hạng nhỏ hơn thì return 1
        elif numbs[c1] < numbs[c2]:
            return -1
        else:
            #thứ hạng =  thì dựa vào chất
            if suits[s1] > suits[s2]:
                return 1
            elif suits[s1] < suits[s2]:
                return -1
        return 0
    
    # hoán vị lá bài lớn hơn lên trước
    for i in range(0, len(a)-1):
        for j in range(i+1, len(a)):
            if cards_comparison(a[i], a[j]) > 0:
                a[i], a[j] = a[j], a[i]

# một phần của thuật toán quicksort, thực hiện việc phân chia mảng dữ liệu thành hai phần
def partition(A, left_index, right_index, B):
    pivot = A[left_index]
    i = left_index + 1
    for j in range(left_index + 1, right_index):
        if A[j] > pivot:
            A[j], A[i] = A[i], A[j]
            B[j], B[i] = B[i], B[j]
            i += 1
    A[left_index], A[i - 1] = A[i - 1], A[left_index]
    B[left_index], B[i - 1] = B[i - 1], B[left_index]
    return i - 1

# Thuật toán quicksort để sắp xếp danh sách
def quick_sort_random(A, left, right, B):
    if left < right:
        pivot = random.randint(left, right - 1)
        A[pivot], A[left] = A[left], A[pivot] # switches the pivot with the left most bound
        B[pivot], B[left] = B[left], B[pivot]
        pivot_index = partition(A, left, right, B)
        quick_sort_random(
            A, left, pivot_index, B
        )  # recursive quicksort to the left of the pivot point
        quick_sort_random(
            A, pivot_index + 1, right, B
        )  # recursive quicksort to the right of the pivot point


dragon_path = path_52cards + "dragon.png"
dragon_img = mpimg.imread(dragon_path)
test_path = "test/"
results_path = "results/"
os.makedirs(test_path, exist_ok=True)
os.makedirs(results_path, exist_ok=True)
error_path = path_52cards + "Error.jpg"
error_img = mpimg.imread(error_path)


# Function chính, thực hiện các bước sau:
    # Nhận input là ảnh chứa bài.
    # Phát hiện và nhận dạng các lá bài trong ảnh (sử dụng model yolo).
    # Tạo danh sách các lá bài.
    # Tạo tất cả các tổ hợp 3 chi từ 13 lá bài.
    # Sử dụng permutations_553 để đánh giá và tính điểm cho mỗi tổ hợp.
    # Sắp xếp các chi theo điểm.
    # Hình dung chi có điểm cao nhất sử dụng Matplotlib.
    # Hiển thị ảnh "dragon" nếu chi mạnh.
    # Trả về ảnh đã được tạo.
def xapxam(img):
    print("\n")
    print("Chào mừng bạn đến với cách hoạt động của chương trình BINH XẬM XÁM")
# 1. Chuẩn bị và tiền xử lý ảnh
    try:
        # tạo file tạm thời lưu trự ảnh cũng như kết quả với mã định danh ngẫu nhiên 
        uuid_generator = str(uuid.uuid4())
        image_path = test_path + uuid_generator + ".jpg"
        result_path = results_path + uuid_generator + ".jpg"
        
        # kiểm tra kích thức ảnh
        width, height = img.size
        if width == 0 or height == 0:
            return error_img
        # Nếu chiều cao ảnh lớn hơn 640 pixel, resize ảnh để giảm kích thước
        if height > 640:
            width = width * 640 // height
            height = 640
            img = img.resize((width, height))

    #     enhancer = ImageEnhance.Brightness(img)
    #     factor = 1.0
    #     img = enhancer.enhance(factor)

        img_save = img.save(image_path)
       
# 2. Nhận dạng bài (predictions)
        # Dự đoán tên các lá bài(Spade, Club, Diamond, Heart) và giá trị 
        # ví dụ: 3 bích là 3S, 4D,7H
        #Pandas: lưu trữ và xử lý dữ liệu dạng bảng [x_min, y_min, x_max, y_max]
        predictions = model(image_path).pandas().xyxy[0]
        print("Kết quả dự đoán\n",predictions)

# 3. Xử lý danh sách bài được nhận dạng (classes):
        # Duyệt qua danh sách các lá bài có và in ra danh sách
        classes = []
        for card_pred in predictions.name:
            if card_pred not in classes:
                classes.append(card_pred)
            # Nếu danh sách đủ 13 lá thì dừng lại
            if len(classes) == 13:
                break
        print("Danh sách 13 lá bài đã nhận diện: \n",classes)
        print("\n")
        # Nếu nhiều hơn 13 lá thì trả về ảnh mặc định
        if len(classes) < 13:
            return error_img
# 4. Tạo danh sách các lá bài (các chi)
        hands = []
        # duyệt qua danh sách bài đã lọc
        # Tách riêng dựa theo giá trị và chất của lá bài
        # cl là viết tắt class
        for cl in classes:
            card = cl[:-1]
            if cl[-1] == "S":
                card = "spade_" + card
            elif cl[-1] == "C":
                card = "club_" + card
            elif cl[-1] == "D":
                card = "diamond_" + card
            elif cl[-1] == "H":
                card = "heart_" + card
            hands.append(card)
        print("Danh sách các lá bài gán lại tên:\n",hands)
        print("\n")

# 5. Sắp xếp các lá bài (sort_n2):
        # dùng hàm sắp xếp các lá bài theo thứ hạng và chất
        # Xếp theo thứ tự từ cao xuống thấp: A, K, Q, J, 10, 9, 8, 7, 6, 5, 4, 3, 2; 
        # Chất theo thứ tự từ cao xuống thấp: Spade, Heart, Diamond, Club.
        sort_n2(hands)
        print("Danh sách đã được sắp xếp theo thứ tự cao đến thấp:\n",hands)
        print("\n")

# 6. Tạo tổ hợp các hand và tính điểm (permutations_553, quick_sort_random)
        # dùng hàm tổ hợp tất cả các trường hợp của 13 lá bài 
        # tính điểm dựa theo loại chi( cù lũ, sảnh,...) và thứ hạng bài
        hands_553, scores_553 = permutations_553(hands)
        #print("diem",scores_553)
        # Sắp xếp tổ hợp theo điểm, điểm cao nhất lên đầu
        quick_sort_random(scores_553, 0, len(scores_553), hands_553)
        print("tổ hợp tất cả các chi gồm ",len(hands_553),".Trong đó bao gồm:\n",hands_553,"\n")
        print("tổ hợp theo điểm của chi gồm ",len(scores_553),".Trong đó bao gồm:\n",scores_553,"\n")

# 7. Lấy chi mạnh nhất và hiển thị kết quả:
        # Lấy ra tổ hợp 3 chi có điểm cao nhất và điểm tương ứng (sc).
        # sc là viết tắt scores
        hand, sc = hands_553[0], scores_553[0]
        # plt là viết tắt của matplotlib.pyplot, một thư viện Python để tạo đồ thị và hình ảnh
        fig, ax = plt.subplots() #tạo ra một hình (figure) và một trục (axes) để vẽ đồ thị.
        fig.set_figwidth(18)
        fig.set_figheight(12)
        # nếu điểm = vô cực(float('inf')) hiển thị sảnh rồng và ảnh con rồng
        if sc == float('inf'):
            ax.imshow(dragon_img)
        # tắt trục hiển thị của figure
        plt.axis("off")
        # Xác định số hàng và số cột để hiển thị các lá bài (3 hàng, 5 cột).
        columns, rows = 5, 3

        # 7.1 2 chi 5
        # Duyệt qua 2 hàng đầu tiên (đại diện cho 2 chi đầu tiên)
        for i in range(2):
            # Lặp qua 5 cột (đại diện cho 5 lá bài trong mỗi chi)
            for j in range(5):
                # lấy tên lá bài từ tổ hợp
                card = hand[i][j]
                # tạo đường dẫn đến file ảnh lá bài
                card_path = path_52cards + prefix + card + suffix
                # đọc ảnh từ file ảnh 
                img = mpimg.imread(card_path)
                # thêm ảnh đó vào figure tại vị trí lưới tương ứng
                fig.add_subplot(rows, columns, i*5 + j + 1)
                # tắt trục hiển thị
                plt.axis("off")
                # Hiển thị hình ảnh
                plt.imshow(img)
        #  7.2 chi 3
        i = 2
        blank_path = path_52cards + "blank.png"
        img = mpimg.imread(blank_path)
        fig.add_subplot(rows, columns, i*5 + 0 + 1)
        plt.axis("off")
        plt.imshow(img)
        fig.add_subplot(rows, columns, i*5 + 4 + 1)
        plt.axis("off")
        plt.imshow(img)
        # Vòng lặp của chi 3 lá
        for j in range(1, 3+1):
            card = hand[i][j-1]
            card_path = path_52cards + prefix + card + suffix
            img = mpimg.imread(card_path)
            fig.add_subplot(rows, columns, i*5 + j + 1)
            plt.axis("off")
            plt.imshow(img)
        # 7.3 lưu kết quả và lưu vào đường dẫn result
        fig.savefig(result_path)
        im = Image.open(result_path)
        pr=predictions
        pr1 = model(image_path).show()
    # 7.4 nếu có lỗi thì hiện ảnh mặc định
    except Exception as e:
        im = mpimg.imread(error_path)
        print(e)
    return pr,im
# Tạo biến demo chạy giao diện Gradio
# pil" (viết tắt của Python Imaging Library)    
demo = gr.Interface(
    xapxam, 
    inputs=[gr.Image(type="pil", label="Tải ảnh 13 lá bài của bạn"),],#gr.Textbox(label="Input 1",placeholder="nhap ten"),gr.Textbox(label="Input 2"),], 
    outputs=[gr.Text(label="Kết quả dự đoán 13 lá bài"),gr.Image(label="Kết quả sắp xếp trả về"),],
    title="Hỗ trợ Binh Xậm Xám",
    examples=[["Images_Card\\random1.jpg"],
              ["Images_Card\\random2.jpg"],
              ["Images_Card\\random3.jpg"],
              ["Images_Card\\random4.jpg"],
              ["Images_Card\\random5.jpg"],
              ],
    description="Đưa ảnh chụp 13 lá bài của bạn vào. Hệ thống sẽ phân tích và sắp xếp bài tối ưu nhất cho bạn"
)
# Khởi chạy giao diện Gradio với tham số Share=True  
# Cho phép chia sẻ giao diện dưới dạng ứng dụng web có thể truy cập từ trình duyệt web
demo.launch(share=True)






