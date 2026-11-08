"""Normalize the retained IEEE Strict OOXML template for python-docx.

The original is never modified. Namespace and universal-measure conversion
changes package syntax, not the template's page or typographic dimensions.
"""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import hashlib
import re
from lxml import etree

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'research/templates/ieee-conference-letter.docx'
SHA256 = '4589779a1a74782dc5a7a657a9907f243c9dd54510cb6c154acaf24cc9079852'


def prepare(destination):
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == SHA256
    namespaces = {
        'wordprocessingml/main': 'wordprocessingml/2006/main',
        'officeDocument/relationships': 'officeDocument/2006/relationships',
        'officeDocument/math': 'officeDocument/2006/math',
        'drawingml/main': 'drawingml/2006/main',
        'drawingml/wordprocessingDrawing': 'drawingml/2006/wordprocessingDrawing',
        'drawingml/picture': 'drawingml/2006/picture',
        'package/relationships': 'package/2006/relationships',
    }
    w = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
    destination.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(SOURCE) as source, ZipFile(destination, 'w', ZIP_DEFLATED) as target:
        for item in source.infolist():
            payload = source.read(item.filename)
            if item.filename.endswith(('.xml', '.rels')):
                text = payload.decode('utf-8')
                for old, new in namespaces.items():
                    text = text.replace('http://purl.oclc.org/ooxml/' + old,
                                        'http://schemas.openxmlformats.org/' + new)
                root = etree.fromstring(text.encode('utf-8'))
                for element in root.iter():
                    for key, value in list(element.attrib.items()):
                        if key.startswith(w) and re.fullmatch(r'-?\d+(?:\.\d+)?pt', value):
                            element.set(key, str(round(float(value[:-2]) * 20)))
                    if element.tag == w + 'ind':
                        for old, new in [('start', 'left'), ('end', 'right')]:
                            if w + old in element.attrib:
                                element.set(w + new, element.attrib.pop(w + old))
                    if element.tag in (w + 'jc', w + 'tab'):
                        value = element.get(w + 'val')
                        if value in ('start', 'end'):
                            element.set(w + 'val', {'start': 'left', 'end': 'right'}[value])
                payload = etree.tostring(root, xml_declaration=True, encoding='UTF-8', standalone=True)
            target.writestr(item, payload)
    return destination
