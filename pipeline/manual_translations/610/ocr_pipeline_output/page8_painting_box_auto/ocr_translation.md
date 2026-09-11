# OCR + Translation: page8_painting_guide.png

Pipeline: OpenCV preprocessing -> Tesseract OCR (`lang=jpn`) -> Google Translate (`deep-translator`). No LLM involved. Auto-detected 42 candidate regions (see detected_regions.png) and OCR'd each separately.

- [region 0] **JA:** BB
  **EN:** BB
  _(conf: 10.0, bbox: [406, 0, 615, 7])_
- [region 1] **JA:** IEああ、BELaELLJ
  **EN:** IE AH, BELaELLJ
  _(conf: 15.3, bbox: [650, 0, 1020, 9])_
- [region 2] **JA:** ※男像はイメージです。
  **EN:** *The male figure is an image.
  _(conf: 78.2, bbox: [2173, 7, 2386, 37])_
- [region 4] **JA:** PAINTING(
  **EN:** Painting (
  _(conf: 16.5, bbox: [428, 64, 999, 142])_
- [region 4] **JA:** 本体などの塗装色
  **EN:** Painting colors such as the main body
  _(conf: 85.3, bbox: [636, 173, 834, 205])_
- [region 4] **JA:** モンザレッド(6596)
  **EN:** Monzaled (6596)
  _(conf: 92.2, bbox: [636, 206, 856, 230])_
- [region 4] **JA:** +ォレッド(3096)+ホワイト(59%6)
  **EN:** + Voled (3096) + White (59% 6)
  _(conf: 66.2, bbox: [636, 234, 983, 263])_
- [region 5] **JA:** TIN(塗装
  **EN:** Tin (Painting
  _(conf: 59.5, bbox: [651, 64, 1076, 142])_
- [region 6] **JA:** よりリアルに仕上けたい方は、下の基本色をご発くたさい。
  **EN:** If you want a more realistic finish, please use the basic colors below.
  _(conf: 78.9, bbox: [1656, 84, 2313, 113])_
- [region 6] **JA:** *括装にはより安全な「水性法料」のご使用をおすすめします。
  **EN:** * We recommend using a safer "water-based fee" for bracketing.
  _(conf: 76.0, bbox: [1656, 116, 2335, 149])_
- [region 6] **JA:** スラスターなどイエロー部の塗装色
  **EN:** Painting color of yellow part such as thruster
  _(conf: 85.9, bbox: [1963, 173, 2355, 207])_
- [region 6] **JA:** ホワイト(7096)+オレンジイエロー(309%)
  **EN:** White (7096) + Orange Yellow (309%)
  _(conf: 87.8, bbox: [1963, 205, 2351, 237])_
- [region 6] **JA:** +オレンジ(少置)
  **EN:** + Orange (Missing)
  _(conf: 80.5, bbox: [1964, 238, 2141, 262])_
- [region 7] **JA:** 四MSN-O06Sシナンジュ指定色
  **EN:** IV MSN-O06S Sinanje designated color
  _(conf: 72.1, bbox: [1073, 92, 1606, 138])_
- [region 7] **JA:** 武器などの沙装色
  **EN:** Sand color for weapons, etc.
  _(conf: 63.3, bbox: [1292, 173, 1492, 204])_
- [region 7] **JA:** +ダークグリーン(3096)
  **EN:** + Dark Green (3096)
  _(conf: 91.6, bbox: [1293, 236, 1562, 261])_
- [region 8] **JA:** |
  **EN:** |
  _(conf: 78.0, bbox: [309, 137, 322, 307])_
- [region 9] **JA:** |
  **EN:** |
  _(conf: 5.0, bbox: [354, 140, 400, 202])_
- [region 10] **JA:** 還還
  **EN:** Give back
  _(conf: 0.0, bbox: [1101, 163, 1181, 205])_
- [region 11] **JA:** ロケット・バズーカの的装色
  **EN:** Rocket Bazooka Targeted Color
  _(conf: 81.6, bbox: [1964, 298, 2266, 329])_
- [region 11] **JA:** グレー(8596)+ブラック(1096)
  **EN:** Grey (8596) + Black (1096)
  _(conf: 85.5, bbox: [1963, 328, 2319, 359])_
- [region 11] **JA:** +ォバープル(596)
  **EN:** + Oberpool (596)
  _(conf: 74.6, bbox: [1963, 357, 2149, 390])_
- [region 12] **JA:** センサーなどの稚装色
  **EN:** Childhood colors such as sensors
  _(conf: 67.7, bbox: [1287, 297, 1533, 334])_
- [region 12] **JA:** クリアー(709%)+クリアブルー(209%)
  **EN:** Clear (709%) + Clear Blue (209%)
  _(conf: 88.5, bbox: [1287, 329, 1716, 356])_
- [region 12] **JA:** +クリアイエロー(1096)
  **EN:** + Clear Yellow (1096)
  _(conf: 91.2, bbox: [1287, 358, 1557, 393])_
- [region 13] **JA:** 人請還
  **EN:** human redemption
  _(conf: 11.0, bbox: [1067, 288, 1186, 343])_
- [region 14] **JA:** 関節などの彼装色
  **EN:** The color of his joints etc.
  _(conf: 60.5, bbox: [635, 308, 834, 344])_
- [region 14] **JA:** グレー(9096)+ブラック(1096)
  **EN:** Grey (9096) + Black (1096)
  _(conf: 91.1, bbox: [635, 338, 992, 373])_
- [region 15] **JA:** し
  **EN:** death
  _(conf: 62.0, bbox: [1753, 296, 1805, 357])_
- [region 17] **JA:** ンーーー
  **EN:** Mmmm.
  _(conf: 48.0, bbox: [425, 419, 560, 453])_
- [region 18] **JA:** アー
  **EN:** A
  _(conf: 33.0, bbox: [1078, 407, 1135, 453])_
- [region 19] **JA:** 袖、胸など装番部の党装色
  **EN:** Party clothing colors on sleeves, chest, etc.
  _(conf: 58.3, bbox: [1291, 423, 1579, 460])_
- [region 19] **JA:** ゴールド(BS9%)
  **EN:** Gold (BS9%)
  _(conf: 77.7, bbox: [1292, 454, 1458, 490])_
- [region 19] **JA:** +クリアイエロー(1596)
  **EN:** +Clear yellow (1596)
  _(conf: 91.6, bbox: [1292, 486, 1562, 511])_
- [region 20] **JA:** 'ロケット・バズーカ、ライトグレー部の小装色
  **EN:** 'Rocket Bazooka, light gray accessory color
  _(conf: 81.3, bbox: [1571, 422, 2380, 454])_
- [region 20] **JA:** ホホワイト(709%)+グレー(3096)
  **EN:** White (709%) + Gray (3096)
  _(conf: 76.1, bbox: [1961, 452, 2319, 484])_
- [region 20] **JA:** +ォバープル(少量)
  **EN:** + Ovapur (small amount)
  _(conf: 78.7, bbox: [1963, 481, 2156, 514])_
- [region 20] **JA:** ※カラー配合は参考値ごあり、画像とカラーガイドの色は軸なる場合があります。
  **EN:** * Reference values are available for color combinations, and the colors of images and color guides may be axial.
  _(conf: 76.9, bbox: [1585, 534, 2374, 563])_
- [region 21] **JA:** 胸などの流半色
  **EN:** Chest, etc.
  _(conf: 36.5, bbox: [634, 424, 808, 473])_
- [region 21] **JA:** ミッドナイトブル(1009%)
  **EN:** Midnight Bull (1009%)
  _(conf: 73.6, bbox: [635, 467, 944, 492])_
- [region 22] **JA:** |
  **EN:** |
  _(conf: 54.0, bbox: [2460, 516, 2478, 599])_
- [region 23] **JA:** FIGUREフル・フロンタル
  **EN:** FIGUREFull Frontal
  _(conf: 91.8, bbox: [409, 613, 861, 649])_
- [region 23] **JA:** 頂、振などの浴徹折
  **EN:** Thorough bathing such as peak and swing
  _(conf: 37.1, bbox: [956, 707, 1117, 749])_
- [region 23] **JA:** ]||革葉色(5096)+ホワイト(5O9)||2)
  **EN:** ] || Leather (5096) + White (5O9) || 2)
  _(conf: 52.6, bbox: [370, 712, 1103, 788])_
- [region 23] **JA:** 原の池代名
  **EN:** Hara no Ike Daimyo
  _(conf: 26.4, bbox: [535, 718, 631, 760])_
- [region 23] **JA:** EE
  **EN:** EEE
  _(conf: 32.0, bbox: [370, 790, 1093, 888])_
- [region 23] **JA:** 彼和の和装色
  **EN:** His Japanese kimono color
  _(conf: 5.2, bbox: [953, 800, 1070, 843])_
- [region 23] **JA:** モンザレッド(1009%)+オレンジイエロー(20%)
  **EN:** Monza Red (1009%) + Orange Yellow (20%)
  _(conf: 82.0, bbox: [538, 841, 1177, 883])_
- [region 23] **JA:** |+ホワイト(少置)
  **EN:** | + White (missing)
  _(conf: 82.1, bbox: [370, 874, 1093, 903])_
- [region 23] **JA:** 靖。"・の治ら
  **EN:** Yasushi. "・Noji et al.
  _(conf: 11.8, bbox: [370, 910, 1088, 947])_
- [region 26] **JA:** 衣、窒などの省該色
  **EN:** Applicable colors for clothing, nitrogen, etc.
  _(conf: 42.8, bbox: [956, 717, 1116, 743])_
- [region 26] **JA:** プラック(8O%)
  **EN:** Plaque (8O%)
  _(conf: 78.8, bbox: [957, 738, 1088, 764])_
- [region 26] **JA:** +ホワイト(1096)
  **EN:** +White (1096)
  _(conf: 68.7, bbox: [956, 759, 1103, 784])_
- [region 26] **JA:** 装生の波閑双
  **EN:** Haisho no Wa Kanju
  _(conf: 10.0, bbox: [953, 811, 1069, 837])_
- [region 26] **JA:** ビまロビピゴ(-91)
  **EN:** Bima Robipigo (-91)
  _(conf: 51.6, bbox: [956, 832, 1093, 857])_
- [region 26] **JA:** +ォオレンジイエロー(206)
  **EN:** + Orange Yellow (206)
  _(conf: 71.7, bbox: [956, 853, 1177, 878])_
- [region 26] **JA:** +ホワイト(少置)
  **EN:** + White (omitted)
  _(conf: 78.9, bbox: [956, 874, 1094, 899])_
- [region 26] **JA:** 早の徒毅色
  **EN:** early daring color
  _(conf: 9.2, bbox: [955, 915, 1051, 941])_
- [region 26] **JA:** ホワイト(7596)
  **EN:** White (7596)
  _(conf: 91.7, bbox: [955, 937, 1088, 957])_
- [region 26] **JA:** つう)
  **EN:** Tu)
  _(conf: 46.0, bbox: [956, 957, 1110, 981])_
- [region 26] **JA:** +昌入色(1096)6
  **EN:** +Choiri color(1096)6
  _(conf: 38.2, bbox: [954, 977, 1192, 1019])_
- [region 27] **JA:** スミ入れしてみよっう!
  **EN:** Let's put Sumi in it!
  _(conf: 87.0, bbox: [1354, 715, 1654, 764])_
- [region 27] **JA:** ガンダムマーカー/スミ
  **EN:** Gundam Marker/Sumi
  _(conf: 94.3, bbox: [1354, 766, 1655, 801])_
- [region 27] **JA:** 入れ用(別売り)などを
  **EN:** Put it in for use (sold separately), etc.
  _(conf: 93.4, bbox: [1353, 803, 1655, 841])_
- [region 27] **JA:** 使用して、キットのスジ
  **EN:** Use the kit stripes
  _(conf: 93.6, bbox: [1353, 840, 1656, 877])_
- [region 27] **JA:** 彫りを塗装することで、
  **EN:** By painting the carvings,
  _(conf: 82.4, bbox: [1353, 878, 1652, 916])_
- [region 27] **JA:** 立体感、リアル感が増し
  **EN:** Increased sense of three-dimensionality and realism
  _(conf: 92.4, bbox: [1353, 914, 1655, 952])_
- [region 27] **JA:** ます。スミ入れするだけ
  **EN:** Masu. Just add ink
  _(conf: 92.1, bbox: [1353, 954, 1654, 989])_
- [region 27] **JA:** で見違えるような仕上
  **EN:** An unmistakable finish
  _(conf: 95.6, bbox: [1353, 990, 1655, 1028])_
- [region 27] **JA:** がりになります。
  **EN:** It will become stiff.
  _(conf: 92.4, bbox: [1354, 1032, 1552, 1059])_
- [region 28] **JA:** |坦茶色(5096)+ホホワイト(5096)[
  **EN:** |Flat brown (5096) + White (5096) [
  _(conf: 55.3, bbox: [527, 714, 854, 784])_
- [region 28] **JA:** 肛の挫徹名
  **EN:** name of anus
  _(conf: 3.4, bbox: [535, 729, 632, 758])_
- [region 29] **JA:** 【before】
  **EN:** [before]
  _(conf: 78.0, bbox: [1745, 1060, 1908, 1098])_
- [region 30] **JA:** 4
  **EN:** 4
  _(conf: 47.0, bbox: [2072, 717, 2186, 835])_
- [region 30] **JA:** ミッ
  **EN:** Mi
  _(conf: 34.0, bbox: [2072, 963, 2349, 1098])_
- [region 32] **JA:** バンツの緒装色
  **EN:** Bantu's dress color
  _(conf: 58.5, bbox: [538, 932, 667, 960])_
- [region 32] **JA:** ホワイト(1009%)
  **EN:** White (1009%)
  _(conf: 80.7, bbox: [538, 955, 682, 973])_
- [region 32] **JA:** ||ウッドブラウン(10096)
  **EN:** || Wood Brown (10096)
  _(conf: 77.7, bbox: [407, 1009, 732, 1085])_
- [region 32] **JA:** プーツの之話双
  **EN:** Tales of the Poots Double
  _(conf: 73.0, bbox: [538, 1025, 667, 1053])_
- [region 33] **JA:** ん県一す|
  **EN:** Nkenichisu |
  _(conf: 40.4, bbox: [1818, 1079, 1908, 1100])_
- [region 34] **JA:** |
  **EN:** |
  _(conf: 33.0, bbox: [359, 1091, 400, 1161])_
- [region 36] **JA:** -|
  **EN:** None
  _(conf: 65.5, bbox: [2386, 1168, 2493, 1297])_
- [region 37] **JA:** |1/1ロロscaleSINANIJU
  **EN:** |1/1 Roro scaleSINANIJU
  _(conf: 77.0, bbox: [314, 1189, 1063, 1280])_
- [region 39] **JA:** ーー
  **EN:** -
  _(conf: 46.0, bbox: [1597, 1216, 1963, 1222])_