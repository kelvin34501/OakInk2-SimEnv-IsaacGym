import re


def to_camel(snake_case):
    return "".join(str.capitalize(el) for el in snake_case.split("_"))


# 预先编译需要使用的正则表达式
uppercase_pattern = re.compile(r"([A-Z]+)([A-Z][a-z])")
camelcase_pattern = re.compile(r"([a-z\d])([A-Z])")


def to_snake_case(camel):
    # 使用已编译的正则表达式处理连续的大写字母
    camel = uppercase_pattern.sub(r"\1_\2", camel)
    # 使用已编译的正则表达式处理普通的CamelCase情况
    camel = camelcase_pattern.sub(r"\1_\2", camel)
    # 转换为小写
    return camel.lower()
