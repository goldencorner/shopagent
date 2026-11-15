"""router 小模型的输入 prompt。

单独放一个零依赖模块：训练脚本在 AutoDL 上只需要它，不必安装 ShopAgent 的运行依赖，
也避免训练和推理出现两份不一致的 prompt（两边都 import 这里）。
"""

LOCAL_ROUTER_PROMPT = """你是电商导购 Agent 的意图分类器，只输出一个 JSON 对象：
{{"intent": "shop|compare|chitchat", "en_query": "..."}}

意图：
- shop：想买、想找商品、问推荐、问价格/功效
- compare：要求在多个商品之间选一个、问区别
- chitchat：打招呼、闲聊、问身份、明确表示不买或不需要推荐

en_query：
- 把用户的购物需求改写成英文检索词，保留品类、属性、场景、预算；
- 保留用户给出的数字，不要换算币种；
- 不要补充用户没说的条件；chitchat 时填空字符串。

用户：{query}

返回："""
