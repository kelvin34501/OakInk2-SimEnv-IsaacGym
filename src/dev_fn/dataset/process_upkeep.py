import os
import re


class ProcessDef:
    sep = ","

    def __init__(self, process_key: str, beg=None, end=None, range_comment=None):
        # split with sep
        _info = process_key.split(self.sep)
        if len(_info) == 1:
            self.process_key = str(_info[0])
            if beg is not None and end is not None:
                self.beg, self.end = beg, end
            else:
                self.beg, self.end = None, None
            self.range_comment = range_comment
        elif len(_info) == 2:
            self.process_key = str(_info[0])
            if beg is not None and end is not None:
                self.beg, self.end = beg, end
            else:
                self.beg, self.end = None, None
            self.range_comment = str(_info[1])
        elif len(_info) == 3:
            self.process_key = str(_info[0])
            self.beg = int(_info[1])
            self.end = int(_info[2])
            self.range_comment = None
        elif len(_info) == 4:
            self.process_key = str(_info[0])
            self.beg = int(_info[1])
            self.end = int(_info[2])
            self.range_comment = str(_info[3])
        else:
            raise ValueError(f"Invalid process key: {process_key}")

    def __repr__(self) -> str:
        has_range_info = not (self.beg is None or self.end is None)
        has_range_comment = self.range_comment is not None

        if has_range_info:
            range_info_str = f"{self.sep} {self.beg:0>6}{self.sep} {self.end:0>6}"
        else:
            range_info_str = ""

        if has_range_comment:
            comment_info_str = f'{self.sep} "{self.range_comment}"'
        else:
            comment_info_str = ""

        return f'ProcessDef("{self.process_key}"{range_info_str}{comment_info_str})'

    def __str__(self) -> str:
        has_range_info = not (self.beg is None or self.end is None)
        has_range_comment = self.range_comment is not None

        if has_range_info:
            range_info_str = f"{self.sep}{self.beg:0>6}{self.sep}{self.end:0>6}"
        else:
            range_info_str = ""

        if has_range_comment:
            comment_info_str = f"{self.sep}{self.range_comment}"
        else:
            comment_info_str = ""

        return f"{self.process_key}{range_info_str}{comment_info_str}"

    def __eq__(self, other) -> bool:
        if not isinstance(other, ProcessDef):
            return False
        return (
            self.process_key == other.process_key
            and self.beg == other.beg
            and self.end == other.end
            and self.range_comment == other.range_comment
        )

    def __hash__(self) -> int:
        return hash((self.process_key, self.beg, self.end, self.range_comment))

    # def ineq relation for sort
    def __lt__(self, other) -> bool:
        if not isinstance(other, ProcessDef):
            raise ValueError(f"Invalid type: {type(other)}")
        # check process_key
        if self.process_key < other.process_key:
            return True
        if self.process_key > other.process_key:
            return False
        # full seq is gt partial seq
        if self.beg is None or self.end is None:
            return False
        if other.beg is None or other.end is None:
            return True
        # beg first
        if self.beg < other.beg:
            return True
        if self.beg > other.beg:
            return False
        # end first
        if self.end < other.end:
            return True
        if self.end > other.end:
            return False
        # range_comment lexical order
        if self.range_comment is None and other.range_comment is not None:
            return True
        if self.range_comment is not None and other.range_comment is None:
            return False
        if self.range_comment is not None and other.range_comment is not None:
            return self.range_comment < other.range_comment
        # equal
        return False

    def to_offset(self):
        _offset = self.process_key.replace("/", "++")
        _res_list = [_offset]

        has_range_info = not (self.beg is None or self.end is None)
        has_range_comment = self.range_comment is not None
        if has_range_info:
            _res_list.append(f"{self.beg:0>6}_{self.end:0>6}")
        if has_range_comment:
            _res_list.append(f"@{self.range_comment}")

        _res = "++".join(_res_list)
        return _res

    match_range_comment = re.compile(r"^@(.*)$")
    match_range_info = re.compile(r"^(\d+)_(\d+)$")

    @staticmethod
    def from_offset(offset_str):
        offset_list = offset_str.split("++")

        # iterate in reverse order, match for range_comment and range_info
        range_comment = None
        range_info = None

        for _rev_offset, _part in enumerate(reversed(offset_list)):
            _offset = len(offset_list) - 1 - _rev_offset
            if range_comment is None:
                _comment_match = ProcessDef.match_range_comment.fullmatch(_part)
                if _comment_match:
                    range_comment = _comment_match.group(1)
                    continue

            if range_info is None:
                _range_info_match = ProcessDef.match_range_info.fullmatch(_part)
                if _range_info_match:
                    _range_info_beg = int(_range_info_match.group(1))
                    _range_info_end = int(_range_info_match.group(2))
                    range_info = (_range_info_beg, _range_info_end)
                    continue

            break

        exit_offset = _offset
        offset_list = offset_list[: exit_offset + 1]

        process_key = "/".join(offset_list)
        range_info_beg = range_info[0] if range_info is not None else None
        range_info_end = range_info[1] if range_info is not None else None

        process_def = ProcessDef(
            process_key=process_key, beg=range_info_beg, end=range_info_end, range_comment=range_comment
        )

        return process_def

