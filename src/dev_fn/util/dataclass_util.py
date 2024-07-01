from dataclasses import dataclass, fields
import typing


@dataclass
class NamedData:
    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def get(self, key, default=None):
        return getattr(self, key, default)

    def __iter__(self):
        return self.values()

    def keys(self):
        keys = [t.name for t in fields(self)]
        return iter(keys)

    def values(self):
        values = [getattr(self, t.name) for t in fields(self)]
        return iter(values)

    def items(self):
        data = [(t.name, getattr(self, t.name)) for t in fields(self)]
        return iter(data)


# class NamedDataConvert:
#     def __init__(self):
#         self.store = {}

#     def reg(self, cls):
#         self.store[cls.__name__] = cls
#         return cls

#     def to_named_data(self, named_data_cls, data):
#         if isinstance(data, dict):
#             _res = {}
#             for k, v in data.items():
#                 _cls = named_data_cls.__annotations__[k]
#                 if not isinstance(_cls, type):
#                     _cls = self.store[_cls]
#                 _res[k] = self.to_named_data(_cls, v)
#             return named_data_cls(**_res)
#         # TODO: list
#         # elif isinstance(data, list):
#         #     elem_type = named_data_cls.__args__[0]
#         #     return [to_named_data(elem_type, elem) for elem in data]
#         else:
#             return data
