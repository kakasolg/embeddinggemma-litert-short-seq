"""Facts for the public demo dataset — every shop, place, person and number here is fictional.

Each item becomes one image. `lines` is the text in reading order (what OCR should roughly return).
`style` picks the renderer in gen_dataset.py; `photo=True` adds camera-like distortion.
Design and results: the repository README ("Find the text in your phone's photos"). Questions live in questions.py and point at these ids.
"""

# ── menus (20) ──────────────────────────────────────────────────────────────
# Long single-column menus (style "menu_screen") keep reading order predictable, so the "deep" items
# (late in the menu, past the first 64 tokens) really are late in the OCR text.

MENUS = [
    dict(id="m01", style="menu_screen", photo=False, lang="ko+en", title="한빛식당 Hanbit Kitchen",
         sub="Korean home cooking · 한식", items=[
             ("김치찌개 Kimchi Jjigae", "16.99"), ("된장찌개 Doenjang Jjigae", "15.99"),
             ("순두부찌개 Soft Tofu Stew", "16.49"), ("제육볶음 Spicy Pork", "18.99"),
             ("불고기 Bulgogi", "22.99"), ("갈비탕 Galbi-tang", "19.99"), ("비빔밥 Bibimbap", "15.99"),
             ("돌솥비빔밥 Stone Pot Bibimbap", "17.49"), ("물냉면 Cold Noodles", "15.49"),
             ("떡볶이 Tteokbokki", "13.99"), ("해물파전 Seafood Pancake", "19.49"),
             ("김치전 Kimchi Pancake", "14.99"), ("잡채 Japchae", "16.99"), ("고등어구이 Grilled Mackerel", "18.49"),
             ("육개장 Yukgaejang", "17.99"), ("감자탕 (2인) Pork Bone Stew for 2", "42.00"), ("공기밥 Rice", "2.50")]),
    dict(id="m02", style="menu_screen", photo=False, lang="en+ko", title="Maru Cafe 마루카페",
         sub="Coffee · Tea · Desserts", items=[
             ("Americano", "4.25"), ("Cafe Latte", "5.25"), ("Vanilla Latte", "5.75"), ("Cold Brew", "5.00"),
             ("Misugaru Latte 미숫가루 라떼", "6.25"), ("Yuja Tea 유자차", "5.50"), ("Jujube Tea 대추차", "5.75"),
             ("Matcha Latte", "6.00"), ("Hojicha Latte", "6.00"), ("Strawberry Milk", "6.50"),
             ("Injeolmi Bingsu 인절미 빙수", "13.50"), ("Mango Bingsu", "14.00"), ("Matcha Cheesecake", "7.25"),
             ("Croffle", "6.75"), ("Butter Bar", "4.50"), ("Black Sesame Affogato", "7.75")]),
    dict(id="m03", style="menu_screen", photo=False, lang="ko", title="소라 순두부",
         sub="순두부 전문 · 매운맛 0~5단계", items=[
             ("기본 순두부", "15.99"), ("해물 순두부", "17.49"), ("소고기 순두부", "16.99"), ("돼지고기 순두부", "16.49"),
             ("버섯 순두부", "15.99"), ("햄 치즈 순두부", "16.49"), ("만두 순두부", "16.49"), ("김치 순두부", "15.99"),
             ("순두부 + 갈비 콤보", "27.99"), ("순두부 + 불고기 콤보", "25.99"), ("순두부 + 고등어 콤보", "24.99"),
             ("계란 추가", "1.00"), ("공기밥 추가", "2.50"), ("굴 순두부 (계절 한정)", "18.99")]),
    dict(id="m04", style="menu_screen", photo=False, lang="en", title="Blue Lantern Bistro",
         sub="Brunch · Sat & Sun 9 AM – 3 PM", items=[
             ("Classic Eggs Benedict", "18.00"), ("Crab Cake Benedict", "24.00"), ("Avocado Toast", "15.00"),
             ("Buttermilk Pancakes", "14.00"), ("Brioche French Toast", "16.00"), ("Shakshuka", "17.00"),
             ("Steak & Eggs", "29.00"), ("Smoked Salmon Bagel", "19.00"), ("Kale Caesar", "14.00"),
             ("Burrata & Peaches", "18.00"), ("Mimosa", "12.00"), ("Fresh Orange Juice", "7.00"),
             ("Seasonal Fruit Bowl", "11.00"), ("Duck Confit Hash", "24.00")]),
    dict(id="m05", style="menu_board", photo=True, lang="en+ko", title="Pine Hill Bakery 파인힐 베이커리",
         sub="Fresh daily", items=[
             ("Salt Bread 소금빵", "3.95"), ("Milk Bread Loaf", "8.50"), ("Croissant", "4.25"),
             ("Pain au Chocolat", "4.75"), ("Cream Cheese Garlic Bread", "6.95"), ("Sweet Potato Bread", "5.25"),
             ("Red Bean Butter Bun", "4.75")]),
    dict(id="m06", style="menu_board", photo=True, lang="en", title="La Brisa Tacos",
         sub="Street tacos · corn tortillas", items=[
             ("Al Pastor Taco", "3.75"), ("Carne Asada Taco", "4.00"), ("Chicken Taco", "3.50"),
             ("Carnitas Taco", "3.75"), ("Fish Taco", "4.50"), ("Elote", "5.00"), ("Horchata", "4.00"),
             ("Birria Quesataco", "5.50")]),
    dict(id="m07", style="menu_screen", photo=False, lang="en", title="Kaze Ramen",
         sub="Tonkotsu · Shoyu · Shio", items=[
             ("Tonkotsu Ramen", "16.50"), ("Spicy Tonkotsu Ramen", "17.00"), ("Black Garlic Tonkotsu", "17.50"),
             ("Shoyu Ramen", "15.50"), ("Miso Ramen", "16.00"), ("Vegetable Ramen", "15.00"), ("Tsukemen", "18.00"),
             ("Gyoza (6)", "8.00"), ("Karaage", "9.50"), ("Chashu Rice Bowl", "8.50"), ("Extra Noodles", "3.00"),
             ("Ajitama Egg", "2.50"), ("Yuzu Shio Ramen", "17.50")]),
    dict(id="m08", style="menu_screen", photo=False, lang="ko+en", title="엠버앤오크 Ember & Oak Korean BBQ",
         sub="무제한 · All You Can Eat · 2시간 이용", items=[
             ("런치 무제한 Lunch AYCE (11–3)", "28.99"), ("디너 A Dinner A", "36.99"), ("프리미엄 Premium", "46.99"),
             ("어린이 (6–10세) Kids", "16.99"), ("우삼겹 Brisket", "포함"), ("양념갈비 Marinated Short Rib", "포함"),
             ("삼겹살 Pork Belly", "포함"), ("항정살 Pork Jowl", "포함"), ("된장찌개 Soybean Stew", "포함"),
             ("계란찜 Steamed Egg", "포함"), ("남긴 고기 1인분당 Leftover meat per portion", "10.00"),
             ("소주 Soju", "14.00"), ("맥주 Beer", "7.00"), ("차돌박이 추가 Extra Chadol", "12.99")]),
    dict(id="m09", style="menu_board", photo=True, lang="en+ko", title="Lotus Pho 연꽃 쌀국수",
         sub="Vietnamese noodle soup", items=[
             ("Pho Tai (rare beef) S/L", "14.50 / 16.50"), ("Pho Ga (chicken) S/L", "13.95 / 15.95"),
             ("Bun Bo Hue", "16.50"), ("Spring Rolls (2)", "7.50"), ("Vietnamese Iced Coffee", "5.50"),
             ("Pho Dac Biet Large", "17.95")]),
    dict(id="m10", style="menu_board", photo=True, lang="en", title="Green Field Juice",
         sub="Cold pressed", items=[
             ("Green Machine", "9.50"), ("Beet It", "9.00"), ("Citrus Glow", "8.50"), ("Acai Bowl", "12.50"),
             ("Protein Smoothie", "10.50"), ("Ginger Shot", "3.50")]),
    dict(id="m11", style="menu_screen", photo=False, lang="ko", title="용문각",
         sub="중화요리 · 배달 가능", items=[
             ("짜장면", "9.99"), ("간짜장", "11.99"), ("짬뽕", "12.99"), ("볶음밥", "12.49"), ("잡채밥", "13.49"),
             ("울면", "12.99"), ("탕수육 (소)", "18.99"), ("탕수육 (대)", "28.99"), ("깐풍기", "26.99"),
             ("라조기", "26.99"), ("군만두", "9.99"), ("유린기", "24.99"), ("삼선짬뽕", "16.99")]),
    dict(id="m12", style="menu_board", photo=True, lang="en", title="Corner Slice",
         sub="NY style pizza", items=[
             ("Cheese Slice", "4.50"), ("Pepperoni Slice", "5.00"), ("Margherita Pie 18\"", "24.00"),
             ("White Pie 18\"", "26.00"), ("Caesar Salad", "9.00"), ("Garlic Knots (6)", "6.50")]),
    dict(id="m13", style="menu_board", photo=True, lang="ko+en", title="눈꽃빙수 Snowflake Bingsu",
         sub="우유 눈꽃 · Milk snow", items=[
             ("인절미 빙수 Injeolmi", "13.50"), ("망고 빙수 Mango", "14.50"), ("딸기 빙수 Strawberry", "14.50"),
             ("팥빙수 Red Bean", "12.50"), ("오레오 빙수 Oreo", "13.50"), ("흑임자 빙수 Black Sesame", "14.50")]),
    dict(id="m14", style="menu_board", photo=True, lang="en", title="Hilltop Burgers",
         sub="Smash burgers", items=[
             ("Single Smash", "9.50"), ("Double Smash", "12.50"), ("Chicken Sandwich", "11.00"),
             ("Veggie Burger", "11.50"), ("Fries", "4.50"), ("Shake", "6.50"), ("Truffle Fries", "7.25")]),
    dict(id="m15", style="menu_board", photo=True, lang="en", title="Nami Sushi",
         sub="Rolls & hand rolls", items=[
             ("California Roll", "9.00"), ("Spicy Tuna Roll", "10.50"), ("Salmon Avocado Roll", "11.00"),
             ("Rainbow Roll", "16.00"), ("Blue Crab Hand Roll", "8.00"), ("Spicy Scallop Hand Roll", "8.50")]),
    dict(id="m16", style="menu_board", photo=True, lang="en", title="Moonrise Donuts",
         sub="Made every morning", items=[
             ("Original Glazed", "2.25"), ("Old Fashioned", "2.50"), ("Maple Bar", "3.25"), ("Boston Cream", "3.50"),
             ("Dozen Assorted", "24.00"), ("Ube Glazed", "3.75")]),
    dict(id="m17", style="menu_board", photo=True, lang="ko", title="서울분식",
         sub="김밥 · 떡볶이 · 라면", items=[
             ("김밥", "7.99"), ("떡볶이", "10.99"), ("라볶이", "12.99"), ("순대", "11.99"), ("어묵", "8.99"),
             ("라면", "9.49"), ("참치김밥", "9.49")]),
    dict(id="m18", style="menu_screen", photo=False, lang="en", title="Siam Garden",
         sub="Thai kitchen · spice level 1–5", items=[
             ("Pad Thai", "15.95"), ("Pad See Ew", "15.95"), ("Drunken Noodles", "16.50"), ("Green Curry", "16.95"),
             ("Red Curry", "16.95"), ("Massaman Curry", "17.50"), ("Tom Yum Soup", "14.50"),
             ("Tom Kha Soup", "14.50"), ("Papaya Salad", "12.95"), ("Larb Gai", "13.95"),
             ("Mango Sticky Rice", "9.95"), ("Thai Iced Tea", "5.50"), ("Khao Soi Chicken", "17.95")]),
    dict(id="m19", style="menu_screen", photo=False, lang="ko", title="다온 찻집",
         sub="전통차 · 한과", items=[
             ("녹차", "5.00"), ("현미녹차", "5.00"), ("유자차", "5.50"), ("생강차", "5.50"), ("대추차", "6.00"),
             ("쌍화차", "6.50"), ("매실차", "5.50"), ("식혜", "5.00"), ("수정과", "5.00"), ("약과 (3개)", "6.00"),
             ("인절미", "6.50"), ("한과 모둠", "12.00"), ("오미자차", "6.50")]),
    dict(id="m20", style="menu_board", photo=True, lang="en", title="Sunny Side Diner",
         sub="All day breakfast", items=[
             ("Two Eggs Any Style", "11.50"), ("Breakfast Burrito", "13.50"), ("Chicken & Waffles", "17.00"),
             ("Short Stack", "9.50"), ("Biscuits & Gravy", "12.00"), ("Loco Moco", "16.75")]),
]

# ── parking (12) ────────────────────────────────────────────────────────────

PARKING = [
    dict(id="p01", style="sign", photo=True, lang="en", color="blue", lines=[
        "MAPLE PLAZA PARKING", "First 90 min FREE with validation", "$3 each additional 30 min",
        "Daily maximum $18", "Validate at participating stores"]),
    dict(id="p02", style="sign", photo=True, lang="en", color="green", lines=[
        "LOT B · HARBOR CENTER", "2 hours free parking", "No overnight parking 2 AM – 5 AM",
        "Violators will be towed at owner's expense"]),
    dict(id="p03", style="sign", photo=True, lang="ko", color="white", lines=[
        "한빛몰 주차 안내", "영수증 지참 시 2시간 무료", "이후 30분당 $2", "지하 2층 전기차 충전소"]),
    dict(id="p04", style="sign", photo=True, lang="en", color="white", lines=[
        "2 HOUR PARKING", "8 AM – 6 PM", "EXCEPT SUNDAYS", "Vehicles with Zone 12 permits exempt"]),
    dict(id="p05", style="sign", photo=True, lang="en", color="black", lines=[
        "VALET PARKING", "$5 + tip", "Self parking not available"]),
    dict(id="p06", style="sign", photo=True, lang="en", color="green", lines=[
        "EV CHARGING · LEVEL P2", "4 hour maximum while charging", "$0.35 per kWh", "Move vehicle when charging is complete"]),
    dict(id="p07", style="sign", photo=True, lang="ko", color="blue", lines=[
        "주차장 이용 시간", "06:00 – 24:00", "24시 이후 출차 불가", "분실 주차권 $25"]),
    dict(id="p08", style="sign", photo=True, lang="en", color="yellow", lines=[
        "MAPLE PLAZA GARAGE", "CLEARANCE 6'8\" (2.03 m)", "Oversized vehicles use surface lot"]),
    dict(id="p09", style="sign", photo=True, lang="en", color="red", lines=[
        "CUSTOMER PARKING ONLY", "15 MIN LOADING ZONE", "Unauthorized vehicles towed"]),
    dict(id="p10", style="sign", photo=True, lang="en", color="blue", lines=[
        "HARBOR CENTER GARAGE", "Weekend flat rate $10", "Saturday & Sunday", "Weekdays $2 per hour"]),
    dict(id="p11", style="sign", photo=True, lang="ko+en", color="black", lines=[
        "Ember & Oak 고객 주차", "3시간 무료 (발렛 제외)", "Customers: 3 hours free", "Valet not included"]),
    dict(id="p12", style="sign", photo=True, lang="en", color="white", lines=[
        "PAY BY PLATE", "Enter your license plate at the kiosk", "First 20 minutes free",
        "Then $2.50 per hour"]),
]

# ── receipts (12) ── totals are computed in gen_dataset.py from items, tax rate and tip ────────────

RECEIPTS = [
    dict(id="r01", style="receipt", photo=True, lang="en", shop="HANBIT KITCHEN", date="09/12/2026 7:42 PM",
         server="Jina", items=[("Kimchi Jjigae", "16.99"), ("Seafood Pancake", "19.49"), ("Rice", "2.50")],
         tax_rate="0.095", tip="7.00"),
    dict(id="r02", style="receipt", photo=True, lang="en", shop="MARU CAFE", date="09/14/2026 2:15 PM",
         server="Theo", items=[("Misugaru Latte", "6.25"), ("Croffle", "6.75"), ("Americano", "4.25")],
         tax_rate="0.095", tip="3.50"),
    dict(id="r03", style="receipt", photo=True, lang="en", shop="BLUE LANTERN BISTRO", date="09/20/2026 11:05 AM",
         server="Ava", items=[("Crab Cake Benedict", "24.00"), ("Shakshuka", "17.00"), ("Mimosa", "12.00"),
                              ("Fresh Orange Juice", "7.00")], tax_rate="0.095", tip="12.00"),
    dict(id="r04", style="receipt", photo=True, lang="en", shop="KAZE RAMEN", date="09/22/2026 8:30 PM",
         server="Ken", items=[("Tonkotsu Ramen", "16.50"), ("Gyoza (6)", "8.00")], tax_rate="0.095", tip="5.00"),
    dict(id="r05", style="receipt", photo=True, lang="en", shop="LA BRISA TACOS", date="09/23/2026 1:10 PM",
         server="Counter", items=[("Al Pastor Taco x3", "11.25"), ("Horchata", "4.00")], tax_rate="0.095", tip="2.00"),
    dict(id="r06", style="receipt", photo=True, lang="en", shop="SIAM GARDEN", date="09/25/2026 6:50 PM",
         server="Nok", items=[("Pad Thai", "15.95"), ("Green Curry", "16.95"), ("Thai Iced Tea", "5.50")],
         tax_rate="0.095", tip="7.50"),
    dict(id="r07", style="receipt", photo=True, lang="en", shop="PINE HILL BAKERY", date="09/27/2026 8:05 AM",
         server="Self", items=[("Salt Bread x2", "7.90"), ("Croissant", "4.25")], tax_rate="0.0", tip="0.00"),
    dict(id="r08", style="receipt", photo=True, lang="en", shop="DRAGON GATE", date="09/28/2026 12:20 PM",
         server="Delivery", items=[("Jjajangmyeon", "9.99"), ("Sweet & Sour Pork (L)", "28.99"),
                                   ("Delivery fee", "4.99")], tax_rate="0.095", tip="5.00"),
    dict(id="r09", style="receipt", photo=True, lang="en", shop="NAMI SUSHI", date="10/02/2026 7:15 PM",
         server="Yuki", items=[("Rainbow Roll", "16.00"), ("Spicy Tuna Roll", "10.50"), ("Miso Soup", "3.50")],
         tax_rate="0.095", tip="6.00"),
    dict(id="r10", style="receipt", photo=True, lang="en", shop="HILLTOP BURGERS", date="10/04/2026 12:40 PM",
         server="Pickup", items=[("Double Smash", "12.50"), ("Truffle Fries", "7.25"), ("Shake", "6.50")],
         tax_rate="0.095", tip="0.00"),
    dict(id="r11", style="receipt", photo=True, lang="en", shop="SNOWFLAKE BINGSU", date="10/05/2026 3:15 PM",
         server="Order A27", items=[("Injeolmi Bingsu", "13.50"), ("Mango Bingsu", "14.50")],
         tax_rate="0.095", tip="2.00"),
    dict(id="r12", style="receipt", photo=True, lang="en", shop="SUNNY SIDE DINER", date="10/11/2026 9:40 AM",
         server="Rosa", items=[("Loco Moco", "16.75"), ("Short Stack", "9.50"), ("Coffee x2", "6.00")],
         tax_rate="0.095", tip="6.50"),
]

# ── notices (12) ────────────────────────────────────────────────────────────

NOTICES = [
    dict(id="n01", style="notice_paper", photo=True, lang="en+ko", lines=[
        "Maru Cafe 마루카페", "FREE WI-FI", "Network: MARU_GUEST", "Password: latte2026",
        "1 hour limit when busy · 혼잡 시 1시간"]),
    dict(id="n02", style="notice_paper", photo=True, lang="ko", lines=[
        "한빛식당 휴무 안내", "10월 9일(금) 한글날", "하루 휴무합니다.", "10월 10일(토)부터 정상 영업합니다."]),
    dict(id="n03", style="notice_screen", photo=False, lang="en", lines=[
        "Pine Hill Bakery", "New hours!", "Starting October 20 we open at 7 AM", "(previously 8 AM)",
        "Closing time stays 6 PM"]),
    dict(id="n04", style="notice_paper", photo=True, lang="ko", lines=[
        "소라 순두부", "브레이크 타임", "15:00 – 17:00", "주문 마감 14:30"]),
    dict(id="n05", style="notice_paper", photo=True, lang="en", lines=[
        "Blue Lantern Bistro", "Corkage fee $20 per bottle", "Limit 2 bottles per table"]),
    dict(id="n06", style="notice_screen", photo=False, lang="ko", lines=[
        "엠버앤오크 생일 혜택", "생일 당일 방문 시 디저트 무료", "신분증을 보여 주세요", "다른 할인과 중복 불가"]),
    dict(id="n07", style="notice_paper", photo=True, lang="en", lines=[
        "Kaze Ramen", "Card minimum $10", "Cash welcome"]),
    dict(id="n08", style="notice_screen", photo=False, lang="en", lines=[
        "Lotus Pho", "Closed Thanksgiving Day", "Thursday, November 26", "Open regular hours Friday"]),
    dict(id="n09", style="notice_paper", photo=True, lang="ko", lines=[
        "용문각 배달 안내", "최소 주문 $25", "배달비 $4.99", "배달 가능 거리 3마일"]),
    dict(id="n10", style="notice_paper", photo=True, lang="en", lines=[
        "Corner Slice", "HAPPY HOUR 3 – 5 PM", "$2 off any slice", "Weekdays only"]),
    dict(id="n11", style="notice_paper", photo=True, lang="ko", lines=[
        "다온 찻집 안내", "어린이 동반 환영합니다", "반려동물은 테라스석만 가능합니다", "실내 동반 불가"]),
    dict(id="n12", style="notice_screen", photo=False, lang="en", lines=[
        "Siam Garden", "Spice levels 1 – 5", "Level 5 is extremely hot", "Level 5 dishes are not refundable"]),
]

# ── booking / order screens (12) ────────────────────────────────────────────

BOOKINGS = [
    dict(id="b01", style="booking", photo=False, lang="en", lines=[
        "Reservation confirmed", "Blue Lantern Bistro", "Sat, Oct 24 · 7:30 PM", "Party of 4",
        "Confirmation #BL-4821"]),
    dict(id="b02", style="booking", photo=False, lang="en", lines=[
        "Reservation confirmed", "Ember & Oak Korean BBQ", "Fri, Oct 30 · 6:00 PM", "Party of 6",
        "2-hour table limit", "Confirmation #EO-1179"]),
    dict(id="b03", style="booking", photo=False, lang="ko", lines=[
        "예약이 확정되었습니다", "한빛식당", "10월 18일(일) 오후 12:30", "3명", "예약번호 HB-2210"]),
    dict(id="b04", style="booking", photo=False, lang="en", lines=[
        "Kaze Ramen waitlist", "You're #5 in line", "Estimated wait 25 min", "We'll text you when your table is ready"]),
    dict(id="b05", style="booking", photo=False, lang="en", lines=[
        "Omakase counter", "Nami Sushi", "Sat, Nov 7 · 8:00 PM", "2 guests", "Deposit paid $50",
        "Confirmation #NS-0073"]),
    dict(id="b06", style="booking", photo=False, lang="en", lines=[
        "Order ready", "Snowflake Bingsu", "Pickup at 3:15 PM", "Order #A27"]),
    dict(id="b07", style="booking", photo=False, lang="en", lines=[
        "Reservation confirmed", "Siam Garden", "Sun, Nov 1 · 5:45 PM", "Party of 5", "Confirmation #SG-5530"]),
    dict(id="b08", style="booking", photo=False, lang="ko", lines=[
        "예약 완료", "다온 찻집", "11월 3일(화) 오후 4:00", "2명", "창가 자리 요청"]),
    dict(id="b09", style="booking", photo=False, lang="en", lines=[
        "Reservation confirmed", "Sunny Side Diner", "Sun, Oct 25 · 9:00 AM", "Party of 4", "High chair requested",
        "Confirmation #SS-0921"]),
    dict(id="b10", style="booking", photo=False, lang="en", lines=[
        "Pickup order", "Hilltop Burgers", "Ready at 12:40 PM", "Order #H-310", "Total $28.74"]),
    dict(id="b11", style="booking", photo=False, lang="en", lines=[
        "Delivery on the way", "Lotus Pho", "Arriving around 7:10 PM", "Order #LP-7782"]),
    dict(id="b12", style="booking", photo=False, lang="en", lines=[
        "Preorder confirmed", "Moonrise Donuts", "Pickup Sat, Oct 31 · 8:00 AM", "2 dozen assorted",
        "Order #MD-1201"]),
]

# ── distractors (12): near-duplicates (hard negatives), unrelated screens, no-text photos ──────────

DISTRACTORS = [
    dict(id="x01", style="menu_board", photo=True, lang="ko", title="한빛분식", sub="분식 · 찌개", items=[
        ("김치찌개", "14.99"), ("부대찌개", "15.99"), ("떡볶이", "9.99"), ("김밥", "6.99"), ("라면", "8.49")]),
    dict(id="x02", style="menu_board", photo=True, lang="en", title="Maru Bakery", sub="Bread & coffee", items=[
        ("Americano", "3.75"), ("Latte", "4.75"), ("Sourdough Loaf", "9.00"), ("Scone", "4.00")]),
    dict(id="x03", style="sign", photo=True, lang="en", color="blue", lines=[
        "OAK PLAZA PARKING", "First 60 min free with validation", "$4 each additional hour"]),
    dict(id="x04", style="widget", photo=False, lang="en", lines=[
        "Weather", "Partly cloudy", "72°F", "Wind 8 mph", "Sunset 6:21 PM"]),
    dict(id="x05", style="widget", photo=False, lang="en", lines=[
        "Now playing", "Late Night Drive", "Neon Harbor", "2:41 / 3:58"]),
    dict(id="x06", style="chat", photo=False, lang="ko", lines=[
        "민호: 이번 주 점심 어디 갈까?", "수아: 국수 먹고 싶어", "민호: 목요일 괜찮아?", "수아: 좋아 12시"]),
    dict(id="x07", style="widget", photo=False, lang="en", lines=[
        "To do", "Buy batteries", "Return library books", "Call the dentist", "Water the plants"]),
    dict(id="x08", style="no_text", photo=True, lang="-", lines=[]),
    dict(id="x09", style="no_text", photo=True, lang="-", lines=[]),
    dict(id="x10", style="widget", photo=False, lang="en", lines=[
        "October 2026", "Mon 26 · Team sync 10:00", "Wed 28 · Gym 18:30", "Sat 31 · Night market"]),
    dict(id="x11", style="menu_board", photo=True, lang="en", title="Blue Harbor Bistro", sub="Seafood", items=[
        ("Clam Chowder", "9.00"), ("Fish & Chips", "19.00"), ("Lobster Roll", "28.00"), ("Oysters (6)", "18.00")]),
    dict(id="x12", style="notice_screen", photo=False, lang="en", lines=[
        "Harbor Night Market", "Saturday, October 31", "6 PM – 11 PM", "Food trucks · Live music"]),
]

ALL = MENUS + PARKING + RECEIPTS + NOTICES + BOOKINGS + DISTRACTORS
