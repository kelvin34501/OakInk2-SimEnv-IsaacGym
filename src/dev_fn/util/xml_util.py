import xml
import xml.etree.ElementTree as ET
from xml.dom import minidom


def _prettify_et(elem):
    """Return a pretty-printed XML string for the Element."""
    if isinstance(elem, str):
        string_mode = True
        elem = ET.XML(elem)
    else:
        string_mode = False
    ET.indent(elem)
    if string_mode:
        res = ET.tostring(elem, encoding="unicode")
        return res
    else:
        return elem


def _prettify_minidom(elem):
    """Return a pretty-printed XML string for the Element."""
    if isinstance(elem, str):
        rough_string = elem
        string_mode = True
    else:
        rough_string = ET.tostring(elem, "utf-8")
        string_mode = False
    reparsed = minidom.parseString(rough_string)
    res = reparsed.toprettyxml(indent="  ", encoding="utf-8").decode("utf-8")
    if string_mode:
        return res
    else:
        return ET.XML(res)


try:
    ET.indent
    prettify = _prettify_et
except AttributeError:
    prettify = _prettify_minidom


def write_xml(filename, root):
    with open(filename, "wb") as f:
        f.write(prettify(root))


def insert_subtree_before(elem_root: ET.Element, elem_handle: ET.Element, sub_tree: ET.Element):
    # Insert the sub-tree before the target tag
    elem_root.insert(list(elem_root).index(elem_handle), sub_tree)


def insert_subtree_after(elem_root: ET.Element, elem_handle: ET.Element, sub_tree: ET.Element):
    # Insert the sub-tree after the target tag
    elem_root.insert(list(elem_root).index(elem_handle) + 1, sub_tree)


def insert_subtree_aschild(elem_root: ET.Element, elem_handle: ET.Element, sub_tree: ET.Element, as_first=False):
    if as_first:
        # Insert the sub-tree as first child of the target tag
        elem_handle.insert(0, sub_tree)
    else:
        # Insert the sub-tree as child of the target tag
        elem_handle.append(sub_tree)


def wrap_in_tag(tag_name: str, elem: ET.Element):
    res = ET.Element(tag_name)
    res.append(elem)
    return res
