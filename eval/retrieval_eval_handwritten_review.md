# 手写检索评测集（草稿 v1）

用途：替代/补充 `retrieval_eval.jsonl` 的「商品反写 query」集合，检验不含标题原词时系统的真实召回。
改完后直接跑：

```bash
python scripts/04_eval_retrieval.py --k 5 --eval eval/retrieval_eval_handwritten_draft.jsonl --store memory
```

线上把 `--store memory` 换成 `--store pg`。`product_title` / `brand` / `price` / `review_notes` 是 review 用的附加字段，评测脚本只读 `query` 和 `relevant_asins`。query 里的预算都刻意压在商品标价之内（如 hw_12 是 $99.95 / 预算 $100），改 query 时记得同步。

字段说明：
- `query`：英文，模拟真实购物者，尽量不含品牌/型号/标题原词；
- `relevant_asins`：当前只标了 1 个目标商品，建议人工补 2-3 个可接受 ASIN；
- `shared_title_words`：query 与商品标题共享的词（自动算的，用来判断泄漏程度）；
- `review_notes`：这条题的设计意图和需要你确认的点。

| id | query（可改） | ASIN | 品牌 | 价格 | 商品标题 | 与标题共享词 |
|---|---|---|---|---|---|---|
| hw_01 | a sleeping bag warm enough for frosty nights but ventable in summer, roomy around the feet, under $250 | B09TDBMRZ6 | MARMOT | $211.95 | MARMOT Never Winter Down Sleeping Bag | bag, sleeping |
| hw_02 | hands-free light for night hiking that throws a beam over 100 meters and runs on AA batteries, around $60 | B012PT1642 | Pelican | $59.48 | Pelican 2780 LED Headlamp (Black) | — |
| hw_03 | small insulated bag that keeps a six-pack cold on day trips and fits inside a backpack, under $15 | B00TUELDQC | Ensign Peak | $10.26 | 6 Pack Nylon Cooler | — |
| hw_04 | clamp-on holder to keep a fishing rod secure on a boat rail, adjustable angle, under $60 | B01BY4HXDM | Moeller Marine | $54.78 | Fishing Rod Holder Set | fishing, holder, rod |
| hw_05 | carry bag for a yoga mat with a zip pocket for keys and phone, machine washable | B0134RRE4U | Gaiam | $26.99 | Gaiam On-The-Go Yoga Mat Carrier | mat, yoga |
| hw_06 | resistance band that attaches at the ankles for sprint and agility drills, comes with drill instructions, under $30 | B0044YSWBW | Rawlings | $24.95 | Rawlings Ankle Resistance Band | band, resistance |
| hw_07 | space-saving adjustable weights for a home gym that go up to about 50 pounds per hand, under $200 | B00MN50JOC | Unipack | $158.82 | Unipack Adjustable Dumbbells | adjustable |
| hw_08 | waterproof covers that keep snow and mud out of my boots when hiking, adjustable, under $30 | B09K3PRR59 | Favuit | $22.82 | Favuit Leg Gaiters Waterproof Men Women Snow Boot Gaiters Hiking Shoe Gaiters Anti-Tear Adjustable Leggings Cover Ankle Leg Guard for Outdoor Hunting Camping Skiing Walking Mountain Climbing | adjustable, hiking, snow, waterproof |
| hw_09 | backpack that carries two rackets plus a separate compartment for dirty shoes, under $60 | B08D5DGFJ6 | K-Swiss | $54.95 | K-Swiss Tennis Backpack | backpack |
| hw_10 | bulk pressureless tennis balls for a ball machine that won't go flat, 60 pack, under $50 | B0BPSVXY1F | Tourna | $47.56 | Tourna Tennis Balls | balls, tennis |
| hw_11 | US Coast Guard approved life jacket for a woman that allows freedom of movement for paddling, under $80 | B0C9XK58SZ | Hyperlite | $74.98 | Hyperlite Logic Life Vest Womens | life |
| hw_12 | travel bag for ski boots with vented side pockets that also holds a helmet and jacket, under $100 | B07C11MG11 | Select Sportbags | $99.95 | Deluxe Ski Boot Bag | bag, ski |
| hw_13 | thin padded gloves for women lifting weights that don't feel bulky, under $25 | B003YHKA06 | HAULIN HOOKS | $19.95 | HAULIN HOOKS Ladies Medium Weight Lifting Gloves | gloves, lifting |
| hw_14 | adjustable speed rope for cardio workouts with comfortable foam handles, under $20 | B0C1P9SB93 | PRCTZ | $14.99 | PRCTZ 9ft PVC Jump Rope, Blue | rope |
| hw_15 | 36-inch foam roller for massaging tight muscles after workouts, medium firmness, under $90 | B01B6KNJ64 | OPTP | $84.0 | LO ROX Aligned Foam Roller by Lauren Roxburgh | foam, roller |
| hw_16 | ultralight canister stove for solo backpacking that boils water in about four minutes and packs into a small case, under $35 | B00DE703MG | Kovea | $29.99 | Kovea Backpackers Camping Stove, Small, Silver | small, stove |
| hw_17 | tiny bike light that clips onto handlebars or seatpost and lasts all week on flashing mode, under $25 | B002UD4TQ6 | KNOG | $19.99 | KNOG Beetle 2-LED Bicycle Light | light |
| hw_18 | size 5 soccer ball with a soft touch for high school practice that keeps its air, under $40 | B09SBS8ZP5 | Select | $34.09 | Select Diamond Soccer Ball, White/Blue/Red V22, Size 5 | ball, soccer |
| hw_19 | 2-liter hydration backpack for trail running that can spray water and attach an inline filter, under $45 | B00870DEVM | Geigerrig | $38.98 | Geigerrig RIG 500 Hydration Pack | hydration |
| hw_20 | four-season backpacking tent for two that holds up in strong wind and snow, has two doors and a vestibule, under $120 | B003RYS7QK | High Peak Outdoors | $99.95 | High Peak Outdoors Klondike Tent | tent |

## 商品信息与设计意图

### hw_01 · B09TDBMRZ6

- **query**：a sleeping bag warm enough for frosty nights but ventable in summer, roomy around the feet, under $250
- **标题**：MARMOT Never Winter Down Sleeping Bag
- **品牌 / 价格 / 类目**：MARMOT / $211.95 / Sports & Outdoors
- **设计意图**：冷天/夏天两用、脚部空间、第二拉链通风是精确定位点；确认 M 码/左右开拉链不构成额外约束
- **描述摘录**：The Never Winter is an ideal summer bag for camping and river trips, but it also has enough water-resistant 650-fill down insulation to keep you warm and protected when frost is in the air. Fold-Down Second Zipper Provides added Ventilation and Easy Access, Main Zipper Anti-Snag Slider reduces Snagging Fold-Down Second Zipper Provides added Ventilation and Easy Access, Main Zipper Anti-Snag Slider reduces Snagging An
- **卖点摘录**：Nylon Made in USA Never Winter-Cilantro Tree Green-REG LEFT Never Winter-Cilantro Tree Green-REG LEFT Marmot Never Winter Sleeping Bag
- **人工修改**：query ▢  relevant_asins 补充 ▢  删除本条 ▢

### hw_02 · B012PT1642

- **query**：hands-free light for night hiking that throws a beam over 100 meters and runs on AA batteries, around $60
- **标题**：Pelican 2780 LED Headlamp (Black)
- **品牌 / 价格 / 类目**：Pelican / $59.48 / Sports & Outdoors
- **设计意图**：亮度 430lm、124m 射程、4×AA 是靠描述区分；若接受其他品牌头灯，可再加可接受 ASIN
- **描述摘录**：Pelican designs and manufactures tough lighting tools used by professionals including firefighters, police, military, aerospace, industrial and consumers. Built Pelican tough, our lights offer the Lumen brightness and safety needed whether in the field or at home. Our line of super bright lights include waterproof lights, tactical lights, safety approved lights for hazardous areas, hands free lighting, right angle li
- **卖点摘录**：Lumens: 430 = high mode; 203 = medium mode; 53 = low mode; Beam Distance: up to 124 meters Run time: 1h30m = high mode; 3h = medium mode; 12h = low mode; Features: 3. 00 inches length, hands free design, Modes = High/Medium/Low/Downcast/Spot/Flashing Batteries
- **人工修改**：query ▢  relevant_asins 补充 ▢  删除本条 ▢

### hw_03 · B00TUELDQC

- **query**：small insulated bag that keeps a six-pack cold on day trips and fits inside a backpack, under $15
- **标题**：6 Pack Nylon Cooler
- **品牌 / 价格 / 类目**：Ensign Peak / $10.26 / Sports & Outdoors
- **设计意图**：6 罐容量与软包形态；竞品很多，建议人工补 1-2 个可接受 ASIN
- **描述摘录**：Bring the cool with you wherever you go with this insulated handy soft cooler. This 6 can cooler is made with 420D Nylon. It's durable and water resistantoutside and inside with a center that's insulated to keep food and beverages cold. Features full zippered top closure and an open front pocket. Cooler Dimensions: 8-1/2” W x 6” H x 6” D
- **卖点摘录**：Holds 6 Cans Easily Durable and Water resistant Full Zippered Top Closure Easy Access Open Front Pocket Measures: 8-1/2” W x 6” H x 6” D
- **人工修改**：query ▢  relevant_asins 补充 ▢  删除本条 ▢

### hw_04 · B01BY4HXDM

- **query**：clamp-on holder to keep a fishing rod secure on a boat rail, adjustable angle, under $60
- **标题**：Fishing Rod Holder Set
- **品牌 / 价格 / 类目**：Moeller Marine / $54.78 / Sports & Outdoors
- **设计意图**：船用夹装、可调角度是题眼；确认不是鱼竿本身
- **描述摘录**：Fishing Rod Holder Set091025-10 Meets the needs of today’s serious fisherman. Available in several mounting options and sizes Additional information Packaged Dimensions 			 14.75"L x 6.75"W x 1.50"H, 374.65mmL x 171.45mmW x 38.10mmH Packaged Weight 			 1.30 lb, .59 kg UPC Code 			 739729002100 Description Meets the needs of today’s serious fisherman. Available in several mounting options and sizes Options include flu
- **卖点摘录**：New
- **人工修改**：query ▢  relevant_asins 补充 ▢  删除本条 ▢

### hw_05 · B0134RRE4U

- **query**：carry bag for a yoga mat with a zip pocket for keys and phone, machine washable
- **标题**：Gaiam On-The-Go Yoga Mat Carrier
- **品牌 / 价格 / 类目**：Gaiam / $26.99 / Sports & Outdoors
- **设计意图**：棉质、可洗、外袋是区分点；价格约束故意不写，考察无价格类 query
- **描述摘录**：Gaiam Pro On-The-Go Mat Bag/h3> The perfect combination of fashion and function, the Gaiam On-The-Go Mat Bag makes it easier to transport your yoga mat to and from practice. Adjustable to fit any size mat and a streamlined design with exterior zip pocket to hold your essentials. Travel with ease. Travel with Gaiam. Features: Streamlined design for easy transport Exterior expandable zip pocket Roomy enough to fit most
- **卖点摘录**：Streamlined, thoughtful design fit to carry nearly any size yoga mat Quick-release magnetic snaps make access and storage quick and easy Internal strap that secures your mat Outer pocket safely holds keys, wallet and phone Machine washable
- **人工修改**：query ▢  relevant_asins 补充 ▢  删除本条 ▢

### hw_06 · B0044YSWBW

- **query**：resistance band that attaches at the ankles for sprint and agility drills, comes with drill instructions, under $30
- **标题**：Rawlings Ankle Resistance Band
- **品牌 / 价格 / 类目**：Rawlings / $24.95 / Sports & Outdoors
- **设计意图**：踝部、速度/敏捷训练、附训练说明；确认不是手部拉力带
- **描述摘录**：Speed is one of the most valuable tools in baseball. This ankle resistance band will allow you to work on your speed and agility by applying resistance to all your important lower body muscles. Features: Ideal resistance level for training your lower body muscles. Includes: Pro drills and instructions. Adjustable Velcro closures for secure fit and versatile for several different exercises.
- **卖点摘录**：Speed training area Ankle resistance band is 12" in length when closed Ideal resistance level for baseball conditioning Includes drills and instructions
- **人工修改**：query ▢  relevant_asins 补充 ▢  删除本条 ▢

### hw_07 · B00MN50JOC

- **query**：space-saving adjustable weights for a home gym that go up to about 50 pounds per hand, under $200
- **标题**：Unipack Adjustable Dumbbells
- **品牌 / 价格 / 类目**：Unipack / $158.82 / Sports & Outdoors
- **设计意图**：可调、单只 52.5lb、家里省空间；这是大件，确认运费/可用性不在评测范围
- **描述摘录**：Not everyone interested in getting a better body through strength training, as you must have enough home space for a dozen or more sets of dumbbells, but in order to get the best results , you need the dumbbells, and, to perform the widest variety of exercises, you need varying weights. This adjustable dumbbell kit is the best and must-have for you! For the 105 lbs dumbbell kits, each includes: 2 pc X dumbbell handle
- **卖点摘录**：Made of high quality cast iron spray painted Easy to set up and no tools required Can be setup in increment of 5 lbs (adjustable from 5 lbs to 52.5 lbs each dumbbell) Solid Chrome Handles and Spin-Lock Star Collars 52.5lbs each, total 105lbs
- **人工修改**：query ▢  relevant_asins 补充 ▢  删除本条 ▢

### hw_08 · B09K3PRR59

- **query**：waterproof covers that keep snow and mud out of my boots when hiking, adjustable, under $30
- **标题**：Favuit Leg Gaiters Waterproof Men Women Snow Boot Gaiters Hiking Shoe Gaiters Anti-Tear Adjustable Leggings Cover Ankle Leg Guard for Outdoor Hunting Camping Skiing Walking Mountain Climbing
- **品牌 / 价格 / 类目**：Favuit / $22.82 / Sports & Outdoors
- **设计意图**：护腿/雪套而非雨裤；确认“waterproof”是否过度提示
- **描述摘录**：Get A Pair of Waterproof Leg Gaiters for Outdoor Activities! The professional’s choice — Favuit Leg Gaiters √ Hiking Waterproof Leg Gaiter √ Skiing Walking Snow Boot Gaiters √ Unisex working hunting feet Gaiter Don't let snow,rain,sand,rocks or dust interfere with your hiking!Whether climbing a mountain covered in snow,weeding in the backyard,walking through muddy and wet bushes or riding a motorcycle in the rain,Fav
- **卖点摘录**：ABRASION RESISTANCE AND DURABLE — Our leg gaiters made of rugged durable nylon Oxford cloth,that can prevent water,snow,dirt and debris from dirtying or damaging your clothes.Excellent in tear & puncture resistance,protects your legs or pants from cuts from ro
- **人工修改**：query ▢  relevant_asins 补充 ▢  删除本条 ▢

### hw_09 · B08D5DGFJ6

- **query**：backpack that carries two rackets plus a separate compartment for dirty shoes, under $60
- **标题**：K-Swiss Tennis Backpack
- **品牌 / 价格 / 类目**：K-Swiss / $54.95 / Sports & Outdoors
- **设计意图**：双拍位、独立鞋仓；注意 query 没有出现 tennis 以免直接抄标题，检验语义召回
- **描述摘录**：The K-Swiss Tennis Backpack is for the player looking for quality, performance, and everyday use. This backpack features a racket specific compartment that will hold 2 rackets with their handles to the side, main compartment with internal organizational pockets within, large pocket on the bottom for shoes or dirty gear, 2 accessory pockets on the front, padded adjustable backpack straps. Dimensions: H21" x D9" x W15"
- **卖点摘录**：Racket specific compartment holds 2 rackets with their handles to the side Main compartment with internal organizational pockets Large pocket on the bottom for shoes or dirty gear 2 accessory pockets on the front Padded adjustable backpack straps
- **人工修改**：query ▢  relevant_asins 补充 ▢  删除本条 ▢

### hw_10 · B0BPSVXY1F

- **query**：bulk pressureless tennis balls for a ball machine that won't go flat, 60 pack, under $50
- **标题**：Tourna Tennis Balls
- **品牌 / 价格 / 类目**：Tourna / $47.56 / Sports & Outdoors
- **设计意图**：无压球、球机、60 只；确认“pressureless”是否属于标题词泄漏（标题没有）
- **描述摘录**：Tourna Pressure less tennis balls never go dead and include 60 bulk tennis balls per pack. They are ideal for practice and for throwing machines. Pressure-less means they never go dead, which makes them great for tennis practice, ball machines, filling up ball baskets and hoppers, or just making sure your pet has hours of fun chasing these balls. They fit Chuck-it style Dog ball launchers and automatic ball launchers
- **卖点摘录**：TOURNA PRESSURLESS BALLS NEVER LOSE THEIR BOUNCE. Regular tennis balls go flat over time, even if you don’t use them. These balls will have the same consistent bounce for the life of the ball. DURABLE! Not all tennis balls are created equal. The extra durable 
- **人工修改**：query ▢  relevant_asins 补充 ▢  删除本条 ▢

### hw_11 · B0C9XK58SZ

- **query**：US Coast Guard approved life jacket for a woman that allows freedom of movement for paddling, under $80
- **标题**：Hyperlite Logic Life Vest Womens
- **品牌 / 价格 / 类目**：Hyperlite / $74.98 / Sports & Outdoors
- **设计意图**：女款、USCG 认证、活动自由度；确认尺码问题不影响相关性
- **描述摘录**：The women?s specific Hyperlite Logic Life Vest combines a stylish look with all the protection you need on the water This US Coast Guard Approved life vest is designed with segmented foam panels that improve the fit while also giving you plenty of mobility It has molded chest cups and secure belt loops with PVC buckles to help dial in the fit and offers a secure hold Add in the improved drainage and even after a jump
- **卖点摘录**：Women's Life Vest
- **人工修改**：query ▢  relevant_asins 补充 ▢  删除本条 ▢

### hw_12 · B07C11MG11

- **query**：travel bag for ski boots with vented side pockets that also holds a helmet and jacket, under $100
- **标题**：Deluxe Ski Boot Bag
- **品牌 / 价格 / 类目**：Select Sportbags / $99.95 / Sports & Outdoors
- **设计意图**：透气靴袋、可装头盔/雪服、背包式；价格刚好 99.95，若人工改价注意边界
- **描述摘录**：Deluxe Triangular Ski Boot Bag / Backpack The bag is designed with a large main compartment and separate side-entry zippered boot pockets. The oversized main compartment will hold just about all your ski gear, including helmet, jacket, gloves, etc. The separate boot pockets, vented with bottom drainage grommets, make it easy to slip your boots inside after a long day on the slopes. Designed with padded backpack strap
- **卖点摘录**：One of the most comfortable and useful ski boot bags on the market.
- **人工修改**：query ▢  relevant_asins 补充 ▢  删除本条 ▢

### hw_13 · B003YHKA06

- **query**：thin padded gloves for women lifting weights that don't feel bulky, under $25
- **标题**：HAULIN HOOKS Ladies Medium Weight Lifting Gloves
- **品牌 / 价格 / 类目**：HAULIN HOOKS / $19.95 / Sports & Outdoors
- **设计意图**：女款、薄、不臃肿；确认与“健身手套”泛品类的区分度
- **描述摘录**：A lady can be tough without giving up her soft touch with our Genuine Haulin Hooks goatskin gloves. Made to fit close with Lycra, these comfortable long wearing gloves have just the right amount of padding to protect, while retaining a trim, non-bulky look and feel. Available in 4 flattering accents, these complimentary gloves are the perfect companion to Haulin' Hooks and straps!
- **卖点摘录**：Genuine Haulin Hooks gloves Form fitted, hand hugging design Buttery soft yet tough Goatskin Perfect for weight lifting, aerobics, cycling, etc. machine washable
- **人工修改**：query ▢  relevant_asins 补充 ▢  删除本条 ▢

### hw_14 · B0C1P9SB93

- **query**：adjustable speed rope for cardio workouts with comfortable foam handles, under $20
- **标题**：PRCTZ 9ft PVC Jump Rope, Blue
- **品牌 / 价格 / 类目**：PRCTZ / $14.99 / Sports & Outdoors
- **设计意图**：可调长度、泡棉手柄；价格 14.99，若改预算注意
- **描述摘录**：The PRCTZ 9ft Jump Rope with Foam Handles is a high-quality jump rope designed to take your cardio workout to the next level. The 9ft length can be easily adjusted to fit users of all heights. The foam handles provide a comfortable grip during long jump rope sessions, reducing hand fatigue and allowing for longer and more intense workouts. The bright blue color adds a pop of style to your workout routine, making it a
- **卖点摘录**：<ul><li>Non-slip EVA foam padded handles help keep your hands dry and comfortable</li> <li>PVC construction offers durability</li> <li>9ft (280cm) long adjustable rope</li> <li>Available in blue and pink color options</li> <li>Great for quick cardio workout on
- **人工修改**：query ▢  relevant_asins 补充 ▢  删除本条 ▢

### hw_15 · B01B6KNJ64

- **query**：36-inch foam roller for massaging tight muscles after workouts, medium firmness, under $90
- **标题**：LO ROX Aligned Foam Roller by Lauren Roxburgh
- **品牌 / 价格 / 类目**：OPTP / $84.0 / Sports & Outdoors
- **设计意图**：36 英寸、中等硬度是区分点；确认长度是否写入标题（标题没有 36）
- **描述摘录**：Developed by fascia and alignment expert Lauren Roxburgh, the LO ROX Aligned Roller is designed to elongate muscles, tone the body and provide regenerative massage.It features a moderate density made from durable EVA foam; perfect for improving blood circulation throughout fascia, skin, muscles and joints.Measures 36" L x 6" diameter.
- **卖点摘录**：Moderate density foam roller designed to tone the body and provide regenerative massage Made from durable EVA foam Developed by fascia and alignment expert Lauren Roxburgh Improves blood circulation and decreases inflammation Measures 36" L x 6" diameter
- **人工修改**：query ▢  relevant_asins 补充 ▢  删除本条 ▢

### hw_16 · B00DE703MG

- **query**：ultralight canister stove for solo backpacking that boils water in about four minutes and packs into a small case, under $35
- **标题**：Kovea Backpackers Camping Stove, Small, Silver
- **品牌 / 价格 / 类目**：Kovea / $29.99 / Sports & Outdoors
- **设计意图**：分体炉头、4 分多钟烧开水、收纳袋；确认“canister”不构成硬约束
- **描述摘录**：The lightweight BACKPACKERS brings you quality at an affordable price. When used with Kovea's isobutane camping gas, 33 ounces of water (4 cups) will boil in just 4 minutes and 17 seconds. Conveniently equipped with a flame control knob and a compact carrying bag, the Backpackers is ideal for solo campers.
- **卖点摘录**：Ultra-light and compact Durable rustproof components Adjustable flame control Folding pan supports for ease of storage Convenient carry case
- **人工修改**：query ▢  relevant_asins 补充 ▢  删除本条 ▢

### hw_17 · B002UD4TQ6

- **query**：tiny bike light that clips onto handlebars or seatpost and lasts all week on flashing mode, under $25
- **标题**：KNOG Beetle 2-LED Bicycle Light
- **品牌 / 价格 / 类目**：KNOG / $19.99 / Sports & Outdoors
- **设计意图**：小、任意安装、闪烁续航；红色尾灯与“车灯”表述需人工确认是否过宽
- **描述摘录**：Packaged up in a little Knog bubble, the new Beetles punch out hard and bright with two super-bright LEDs. Features: New circuitry and two brighter, energy efficient LEDs. 25 hours of constant light and 130 hours of flashing. Attractive clear retail packaging doubles as a protective case for the consumer with a removable peg card. Fits around any handlebar, seatpost, frame tube, or whatever other odd shape your twist
- **卖点摘录**：Battery Type: 2 CR2032, Bulb Color: Red, Light Color: Black, Mount Location: Universal
- **人工修改**：query ▢  relevant_asins 补充 ▢  删除本条 ▢

### hw_18 · B09SBS8ZP5

- **query**：size 5 soccer ball with a soft touch for high school practice that keeps its air, under $40
- **标题**：Select Diamond Soccer Ball, White/Blue/Red V22, Size 5
- **品牌 / 价格 / 类目**：Select / $34.09 / Sports & Outdoors
- **设计意图**：5 号、软触感、保气；确认“high school”与俱乐部/高中场景匹配
- **描述摘录**：The Diamond soccer ball is made of a strong and soft TPU film with a Laminated TPE foam for an extra soft touch. The specially developed underlining and butyl bladder gives the ball excellent air retention. This construction makes for a great practice or match ball at the high school and Club level.
- **卖点摘录**：PREMIUM QUALITY SOCCER BALL: Made with a very strong and durable TPU film with lamination of 2mm TPE foam and a specially constructed underlining with butyl bladder HAND-SEWN CRAFTSMANSHIP: The 32-panel hand-sewn technology ensures an optimal and true flight f
- **人工修改**：query ▢  relevant_asins 补充 ▢  删除本条 ▢

### hw_19 · B00870DEVM

- **query**：2-liter hydration backpack for trail running that can spray water and attach an inline filter, under $45
- **标题**：Geigerrig RIG 500 Hydration Pack
- **品牌 / 价格 / 类目**：Geigerrig / $38.98 / Sports & Outdoors
- **设计意图**：2L、可喷水、可接滤芯；这是较窄的功能点，确认不是普通水袋包
- **描述摘录**：500 Cubic Inches of highly durable, ballistic pack capacity. Reversible, dishwasher safe, pressurized hydration reservoir. Filter Ready. 70 Oz. The simple spray action of the Geigerrig hydration system enables the water to be sprayed for cooling, cleaning, and sharing. The spray action of Geigerrig also enables the use of a Geigerrig plug-n-play, in-line water filter for on-the-go filtration of water found along the 
- **卖点摘录**：Sprays, Filter Ready, 2 Liter (70 oz.) Hydration Engine, Reversable Dishwasher Safe Reservoir 500 Cubic Inches of Pack Capacity. Light Weight, Removable, Internal Pack Frame. Made from heavy duty Ballistic Nylon, padded pack straps, heavily padded and ventilat
- **人工修改**：query ▢  relevant_asins 补充 ▢  删除本条 ▢

### hw_20 · B003RYS7QK

- **query**：four-season backpacking tent for two that holds up in strong wind and snow, has two doors and a vestibule, under $120
- **标题**：High Peak Outdoors Klondike Tent
- **品牌 / 价格 / 类目**：High Peak Outdoors / $99.95 / Sports & Outdoors
- **设计意图**：四季、抗风抗雪、双门门厅、约 6.8lb；确认“for two”与 85x61 英寸相符
- **描述摘录**：This is the ULTIMATE 4 season backpacking tent. Nothing compares to these features or quality in this price range. This tent has been tested in 60 MPH winds - no problems; this has been tested with more than 2 feet of snow fall - no problems, this has been tested in a super-heavy rain storm - no problem, this has been tested in Joshua Tree National Park in 110 degree heat - no problem (but it was hot) and tested in -
- **卖点摘录**：Full Fly Cover Large Vestibule Two full doors 85 inches long x 61 inches wide x 45 inches tall
- **人工修改**：query ▢  relevant_asins 补充 ▢  删除本条 ▢

## 基线结果（2026-11-10，pg 后端，k=5）

```bash
python scripts/04_eval_retrieval.py --k 5 \
  --eval eval/retrieval_eval_handwritten_draft.jsonl --store pg
```

| 模式 | Recall@5 | MRR | nDCG@5 |
|---|---|---|---|
| bm25 | 0.4000 | 0.3125 | 0.3346 |
| vector | 0.1500 | 0.0542 | 0.0781 |
| hybrid (RRF) | 0.2500 | 0.2125 | 0.2215 |
| hybrid_rerank | **0.4500** | **0.3600** | **0.3824** |

对照：商品反写生成的 `retrieval_eval.jsonl` 上 hybrid_rerank 是 R@5 0.90 / MRR 0.7667。这组手写集只有 0.45，差距主要来自生成集复用了商品标题词汇；0.45 更接近「没见过的问法」下的真实水平（也说明目前系统对同义改写比较敏感）。

每条目标在各通道的排名（— = 该通道 top-50 里没有它；融合排名 >20 会被精排截断）：

| id | 目标 ASIN | BM25 | 向量 | 融合 | 精排 | top-5 |
|---|---|---|---|---|---|---|
| hw_01 | B09TDBMRZ6 | 9 | — | 19 | 2 | ✅ |
| hw_02 | B012PT1642 | 14 | — | 26 | — | ❌ |
| hw_03 | B00TUELDQC | — | — | — | — | ❌ |
| hw_04 | B01BY4HXDM | — | 21 | 28 | — | ❌ |
| hw_05 | B0134RRE4U | 1 | — | 17 | 1 | ✅ |
| hw_06 | B0044YSWBW | 1 | 7 | 1 | 1 | ✅ |
| hw_07 | B00MN50JOC | — | — | — | — | ❌ |
| hw_08 | B09K3PRR59 | — | 29 | 44 | — | ❌ |
| hw_09 | B08D5DGFJ6 | 1 | 4 | 1 | 1 | ✅ |
| hw_10 | B0BPSVXY1F | 1 | 2 | 1 | 1 | ✅ |
| hw_11 | B0C9XK58SZ | — | 7 | 25 | — | ❌ |
| hw_12 | B07C11MG11 | 1 | 3 | 1 | 1 | ✅ |
| hw_13 | B003YHKA06 | 28 | — | 40 | — | ❌ |
| hw_14 | B0C1P9SB93 | 2 | — | 10 | 2 | ✅ |
| hw_15 | B01B6KNJ64 | 33 | — | 43 | — | ❌ |
| hw_16 | B00DE703MG | 24 | 40 | 10 | 1 | ✅ |
| hw_17 | B002UD4TQ6 | — | — | — | — | ❌ |
| hw_18 | B09SBS8ZP5 | 4 | 8 | 11 | 7 | ❌ |
| hw_19 | B00870DEVM | 2 | — | 18 | 14 | ❌ |
| hw_20 | B003RYS7QK | 7 | 7 | 4 | 5 | ✅ |

### 怎么理解这 9/20

失败分三类，改 query 时可以区别对待：

1. **召回池就没进**（hw_03 / hw_07 / hw_17）：两条通道 top-50 都没有目标。hw_03 的 "six-pack cold"、hw_07 的 "weights"、hw_17 的 "bike light" 和商品标题用词（cooler / dumbbells / bicycle light）差得远，这类最能暴露同义改写问题。
2. **进了融合池但被 top-20 截断**（hw_02 融合 26、hw_04 28、hw_08 44、hw_11 25、hw_13 40、hw_15 43）：候选池里有正确答案，是融合排序/截断的问题，适合作为调参目标（加权 RRF、放大精排候选数）。
3. **进了精排但排在 5 名外**（hw_18 第 7、hw_19 第 14）：精排没把目标顶上来，可能是 query 属性过多（hw_19 的喷水 + 滤芯）让别的商品更「像」。

人工修改建议：

- 想保留难度就原样保留；hw_03/07/17 属于「合理问法但系统接不住」，是很好的改进靶子；
- 想把指标做成可对外汇报的基线，可以在 query 里加回品类词（headlamp、cooler、dumbbells、bike light），并给 `relevant_asins` 补 2–3 个可接受商品（冷却袋、哑铃、骑行灯、泡沫轴都有大量等价品）；
- 每条通道排名原始数据在 `eval/handwritten_baseline.json`。
