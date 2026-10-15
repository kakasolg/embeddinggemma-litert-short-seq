"""Questions and fixed answers for the demo dataset (written before any model runs).

gold: image ids that answer the question ([] = the answer is not in the dataset; the right behaviour is to abstain).
answer: canonical answer. accept: substrings, any of which (case- and space-insensitive) counts as a correct answer text.
"@r01.total" style answers are filled from the receipt arithmetic in gen_dataset.py.
type: lookup · paraphrase · crosslingual · numeric · deep (fact sits late in a long menu) · none.
"""

Q = [
    # ── lookup: the words in the question appear in the image ──
    dict(id="L01", type="lookup", lang="en", q="Maru Cafe Wi-Fi password", gold=["n01"], answer="latte2026", accept=["latte2026"]),
    dict(id="L02", type="lookup", lang="en", q="Blue Lantern Bistro confirmation number", gold=["b01"], answer="BL-4821", accept=["BL-4821"]),
    dict(id="L03", type="lookup", lang="ko", q="한빛식당 휴무일", gold=["n02"], answer="10월 9일(금)", accept=["10월 9일", "Oct 9"]),
    dict(id="L04", type="lookup", lang="en", q="Lot B overnight parking rule", gold=["p02"], answer="No overnight parking 2 AM – 5 AM", accept=["2 AM", "2AM"]),
    dict(id="L05", type="lookup", lang="en", q="Kaze Ramen card minimum", gold=["n07"], answer="$10", accept=["$10", "10 dollars"]),
    dict(id="L06", type="lookup", lang="ko", q="용문각 배달 최소 주문 금액", gold=["n09"], answer="$25", accept=["$25", "25달러"]),
    dict(id="L07", type="lookup", lang="en", q="Pine Hill Bakery opening time from October 20", gold=["n03"], answer="7 AM", accept=["7 AM", "7AM", "7:00"]),
    dict(id="L08", type="lookup", lang="en", q="Ember & Oak reservation party size", gold=["b02"], answer="6", accept=["6", "six"]),
    dict(id="L09", type="lookup", lang="en", q="Maple Plaza garage clearance height", gold=["p08"], answer="6'8\" (2.03 m)", accept=["6'8", "2.03"]),
    dict(id="L10", type="lookup", lang="en", q="Corner Slice happy hour", gold=["n10"], answer="3 – 5 PM, $2 off any slice (weekdays)", accept=["3", "$2"]),
    dict(id="L11", type="lookup", lang="en", q="Nami Sushi omakase deposit", gold=["b05"], answer="$50", accept=["$50"]),
    dict(id="L12", type="lookup", lang="en", q="Siam Garden level 5 refund policy", gold=["n12"], answer="Level 5 dishes are not refundable", accept=["not refundable", "no refund", "환불 불가"]),

    # ── paraphrase: same language, different words ──
    dict(id="P01", type="paraphrase", lang="en", q="How long can I park for free at Maple Plaza if a store stamps my ticket?", gold=["p01"], answer="90 minutes", accept=["90"]),
    dict(id="P02", type="paraphrase", lang="en", q="What time do we need to be at the Korean BBQ place on Friday?", gold=["b02"], answer="6:00 PM", accept=["6:00", "6 PM", "6PM"]),
    dict(id="P03", type="paraphrase", lang="en", q="How long can my electric car stay at the charger?", gold=["p06"], answer="4 hours", accept=["4 hour", "4시간"]),
    dict(id="P04", type="paraphrase", lang="en", q="What's the internet login at the coffee shop?", gold=["n01"], answer="MARU_GUEST / latte2026", accept=["MARU_GUEST", "latte2026"]),
    dict(id="P05", type="paraphrase", lang="en", q="Can I bring my own wine to Blue Lantern and what does it cost?", gold=["n05"], answer="$20 per bottle (max 2)", accept=["$20"]),
    dict(id="P06", type="paraphrase", lang="en", q="Is the pho place open on Thanksgiving?", gold=["n08"], answer="Closed (Thursday, Nov 26)", accept=["closed", "휴무"]),
    dict(id="P07", type="paraphrase", lang="en", q="Is my truck too tall for the Maple Plaza garage?", gold=["p08"], answer="Clearance 6'8\" (2.03 m)", accept=["6'8", "2.03"]),
    dict(id="P08", type="paraphrase", lang="en", q="When can I pick up the donuts I ordered ahead?", gold=["b12"], answer="Sat, Oct 31 · 8:00 AM", accept=["8:00", "8 AM", "Oct 31"]),
    dict(id="P09", type="paraphrase", lang="en", q="How long until we get a seat at the ramen shop?", gold=["b04"], answer="About 25 min (#5 in line)", accept=["25"]),
    dict(id="P10", type="paraphrase", lang="en", q="What time should we show up for the sushi counter dinner?", gold=["b05"], answer="Sat, Nov 7 · 8:00 PM", accept=["8:00", "8 PM"]),
    dict(id="P11", type="paraphrase", lang="ko", q="고기 남기면 돈 더 내야 해?", gold=["m08"], answer="1인분당 $10", accept=["$10", "10.00", "10달러"]),
    dict(id="P12", type="paraphrase", lang="ko", q="생일에 가면 뭐 공짜로 주는 데 있었는데", gold=["n06"], answer="엠버앤오크: 생일 당일 디저트 무료(신분증)", accept=["디저트", "dessert"]),
    dict(id="P13", type="paraphrase", lang="ko", q="중국집 배달시키면 배달료 얼마야?", gold=["n09"], answer="$4.99", accept=["4.99"]),
    dict(id="P14", type="paraphrase", lang="ko", q="밤 몇 시 전에 주차장에서 차 빼야 돼?", gold=["p07"], answer="24시(자정) 전 — 이후 출차 불가", accept=["24", "자정", "midnight"]),
    dict(id="P15", type="paraphrase", lang="ko", q="순두부집 오후에 쉬는 시간 언제야?", gold=["n04"], answer="15:00 – 17:00", accept=["15:00", "3시", "3 PM"]),

    # ── cross-lingual: question and image are in different languages ──
    dict(id="C01", type="crosslingual", lang="ko", q="메이플 플라자 주차 하루 최대 요금", gold=["p01"], answer="$18", accept=["$18", "18달러"]),
    dict(id="C02", type="crosslingual", lang="ko", q="블루 랜턴 예약 몇 명이야?", gold=["b01"], answer="4명", accept=["4", "four"]),
    dict(id="C03", type="crosslingual", lang="ko", q="아기 의자 부탁한 아침 식당 예약 몇 시야?", gold=["b09"], answer="9:00 AM (Sun, Oct 25)", accept=["9:00", "9 AM", "오전 9"]),
    dict(id="C04", type="crosslingual", lang="ko", q="태국 식당 매운맛 5단계는 환불돼?", gold=["n12"], answer="안 됨 (not refundable)", accept=["not refundable", "환불 안", "환불 불가", "안 돼"]),
    dict(id="C05", type="crosslingual", lang="ko", q="카페 와이파이 이름 뭐였지?", gold=["n01"], answer="MARU_GUEST", accept=["MARU_GUEST"]),
    dict(id="C06", type="crosslingual", lang="en", q="Can I take my dog into Daon Tea Room?", gold=["n11"], answer="Terrace seats only, not inside", accept=["terrace", "테라스"]),
    dict(id="C07", type="crosslingual", lang="en", q="Hanbit Mall parking: how long is free with a receipt?", gold=["p03"], answer="2 hours", accept=["2 hour", "2시간"]),
    dict(id="C08", type="crosslingual", lang="en", q="Which day in October is Hanbit Kitchen closed?", gold=["n02"], answer="Oct 9 (Fri)", accept=["10월 9일", "Oct 9", "October 9"]),
    dict(id="C09", type="crosslingual", lang="en", q="How many people is my Hanbit Kitchen reservation for?", gold=["b03"], answer="3", accept=["3", "three"]),
    dict(id="C10", type="crosslingual", lang="en", q="What's the delivery fee at Dragon Gate?", gold=["n09"], answer="$4.99", accept=["4.99"]),

    # ── numeric: the answer is a number read off the image ──
    dict(id="N01", type="numeric", lang="en", q="Misugaru latte price at Maru Cafe", gold=["m02"], answer="$6.25", accept=["6.25"]),
    dict(id="N02", type="numeric", lang="ko", q="용문각 탕수육 대 가격", gold=["m11"], answer="$28.99", accept=["28.99"]),
    dict(id="N03", type="numeric", lang="en", q="Ember & Oak dinner all-you-can-eat price", gold=["m08"], answer="$36.99 (Dinner A)", accept=["36.99"]),
    dict(id="N04", type="numeric", lang="en", q="What was the total on my Hanbit Kitchen receipt?", gold=["r01"], answer="@r01.total", accept=["@r01.total"]),
    dict(id="N05", type="numeric", lang="en", q="How much tip did I leave at Maru Cafe?", gold=["r02"], answer="@r02.tip", accept=["@r02.tip"]),
    dict(id="N06", type="numeric", lang="en", q="Al pastor taco price at La Brisa", gold=["m06"], answer="$3.75", accept=["3.75"]),
    dict(id="N07", type="numeric", lang="en", q="Total for the Hilltop Burgers pickup order", gold=["b10", "r10"], answer="$28.74", accept=["28.74"]),
    dict(id="N08", type="numeric", lang="en", q="EV charging price per kWh", gold=["p06"], answer="$0.35", accept=["0.35"]),
    dict(id="N09", type="numeric", lang="en", q="How much tax was on the Blue Lantern brunch receipt?", gold=["r03"], answer="@r03.tax", accept=["@r03.tax"]),
    dict(id="N10", type="numeric", lang="ko", q="서울분식 김밥 얼마?", gold=["m17"], answer="$7.99", accept=["7.99"]),

    # ── deep: the fact is near the end of a long single-column menu (past the first 64 tokens) ──
    dict(id="D01", type="deep", lang="en", q="Black sesame affogato price at Maru Cafe", gold=["m02"], answer="$7.75", accept=["7.75"]),
    dict(id="D02", type="deep", lang="ko", q="소라 순두부 굴 순두부 가격", gold=["m03"], answer="$18.99 (계절 한정)", accept=["18.99"]),
    dict(id="D03", type="deep", lang="en", q="Duck confit hash at Blue Lantern", gold=["m04"], answer="$24.00", accept=["24"]),
    dict(id="D04", type="deep", lang="ko", q="엠버앤오크 차돌박이 추가 가격", gold=["m08"], answer="$12.99", accept=["12.99"]),
    dict(id="D05", type="deep", lang="ko", q="용문각 삼선짬뽕 얼마야", gold=["m11"], answer="$16.99", accept=["16.99"]),
    dict(id="D06", type="deep", lang="en", q="Yuzu shio ramen price at Kaze", gold=["m07"], answer="$17.50", accept=["17.50", "17.5"]),
    dict(id="D07", type="deep", lang="en", q="Khao soi at Siam Garden — how much?", gold=["m18"], answer="$17.95", accept=["17.95"]),
    dict(id="D08", type="deep", lang="ko", q="다온 찻집 오미자차 가격", gold=["m19"], answer="$6.50", accept=["6.50", "6.5"]),

    # ── none: not in the dataset — the right answer is "not found" ──
    dict(id="A01", type="none", lang="en", q="Hanbit Kitchen Wi-Fi password", gold=[], answer="(not in photos)", accept=[]),
    dict(id="A02", type="none", lang="en", q="Does Corner Slice validate parking?", gold=[], answer="(not in photos)", accept=[]),
    dict(id="A03", type="none", lang="en", q="Nami Sushi corkage fee", gold=[], answer="(not in photos)", accept=[]),
    dict(id="A04", type="none", lang="ko", q="서울분식 예약 번호", gold=[], answer="(not in photos)", accept=[]),
    dict(id="A05", type="none", lang="en", q="Moonrise Donuts happy hour", gold=[], answer="(not in photos)", accept=[]),
]
