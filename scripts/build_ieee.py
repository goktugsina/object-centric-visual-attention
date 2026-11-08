"""Create the IEEE conference DOCX from its retained official template."""
from copy import deepcopy
from pathlib import Path
import re
import json
from zipfile import ZipFile, ZIP_DEFLATED
from docx import Document
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH as Align, WD_TAB_ALIGNMENT as Tab
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from PIL import Image
from prepare_ieee_template import prepare

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/docx/object_attention_paper.docx'
template=prepare(ROOT/'tmp/ieee/working-template.docx')
doc=Document(template)
title_section=deepcopy(doc.sections[0]._sectPr)
body_section=deepcopy(doc.sections[5]._sectPr)
for section in (title_section,body_section):
    for node in list(section):
        if node.tag in (qn('w:headerReference'),qn('w:footerReference'),qn('w:titlePg')):section.remove(node)
doc.element.body.clear();doc.element.body.append(body_section)
WIDTH=(612-2*45.35-18)/2/72
doc.core_properties.title='Robustness of Object Attention Prediction With Annotated and Detected Objects'
author_details=json.loads((ROOT/'paper/submission_details.json').read_text(encoding='utf-8'))
doc.core_properties.author=author_details['author_name']
doc.core_properties.subject='OSIE and public EMOd empirical study in IEEE conference format'
doc.core_properties.comments=''

def no_numbering(p):
    p._p.get_or_add_pPr().get_or_add_numPr().get_or_add_numId().val=0

def paragraph(text='',style='Body Text'):
    p=doc.add_paragraph(style=style);p.paragraph_format.widow_control=True
    for bit in re.split(r'(\b[nqpa]_ij\b)',text):
        if re.fullmatch(r'[nqpa]_ij',bit):
            p.add_run(bit[0]).italic=True
            r=p.add_run('ij');r.italic=True;r.font.subscript=True
        else:p.add_run(bit)
    return p

def table(lines):
    rows=[[c.strip() for c in line.strip('|').split('|')] for line in lines]
    rows=[r for r in rows if not all(re.fullmatch(r'[-: ]+',c) for c in r)]
    count=len(rows[0]);widths=[1.13,.57,.57,.65,WIDTH-2.92] if count==5 else [1.65,.55,WIDTH-2.2]
    t=doc.add_table(rows=0,cols=count);t.alignment=WD_TABLE_ALIGNMENT.CENTER;t.autofit=False
    for col,width in zip(t.columns,widths):col.width=Inches(width)
    borders=OxmlElement('w:tblBorders')
    for side in ('top','left','bottom','right','insideH','insideV'):
        e=OxmlElement('w:'+side)
        for k,v in [('val','single'),('sz','4'),('color','000000')]:e.set(qn('w:'+k),v)
        borders.append(e)
    t._tbl.tblPr.append(borders)
    for ri,row in enumerate(rows):
        cells=t.add_row().cells
        for ci,text in enumerate(row):
            cell=cells[ci];cell.width=Inches(widths[ci]);cell.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
            p=cell.paragraphs[0];p.style=doc.styles['Body Text']
            f=p.paragraph_format;f.first_line_indent=Pt(0);f.space_before=Pt(2);f.space_after=Pt(2);f.line_spacing=1
            f.keep_with_next=ri<len(rows)-1
            p.alignment=Align.LEFT if ci==0 or count==3 else Align.CENTER
            r=p.add_run(text);r.font.size=Pt(8);r.bold=ri==0
            margins=OxmlElement('w:tcMar')
            for side in ('left','right'):
                e=OxmlElement('w:'+side);e.set(qn('w:w'),'45');e.set(qn('w:type'),'dxa');margins.append(e)
            cell._tc.get_or_add_tcPr().append(margins)
        props=t.rows[ri]._tr.get_or_add_trPr();props.append(OxmlElement('w:cantSplit'))
        if ri==0:props.append(OxmlElement('w:tblHeader'))
    p=paragraph('','Normal');p.paragraph_format.space_after=Pt(3);p.paragraph_format.line_spacing=Pt(1);p.add_run().font.size=Pt(1)

lines=(ROOT/'paper/manuscript.md').read_text(encoding='utf-8').splitlines()
i=0;references=False;abstract_next=False;title_done=False;section_num=0
roman=['I','II','III','IV','V','VI','VII','VIII','IX']
while i<len(lines):
    line=lines[i].strip();i+=1
    if not line:continue
    if line.startswith('# '):paragraph(doc.core_properties.title,'paper title');continue
    if not title_done:
        p=paragraph(line,'Author');p.paragraph_format.space_after=Pt(3)
        while i<len(lines) and lines[i].strip()!='### Abstract':
            author_line=lines[i].strip();i+=1
            if author_line:
                p=paragraph(author_line,'Normal');p.alignment=Align.CENTER
                p.paragraph_format.space_after=Pt(3)
                for run in p.runs:run.font.name='Times New Roman';run.font.size=Pt(10)
        p.paragraph_format.space_after=Pt(14)
        p=paragraph('','Normal');p.paragraph_format.space_after=Pt(0);p.paragraph_format.line_spacing=Pt(1);p.add_run().font.size=Pt(1)
        p._p.get_or_add_pPr().append(title_section);title_done=True;continue
    if line=='### Abstract':abstract_next=True;continue
    if abstract_next:paragraph('Abstract—'+line,'Abstract');abstract_next=False;continue
    if line.startswith('Keywords: '):paragraph(line.replace('Keywords: ','Keywords—'),'Keywords');continue
    if line.startswith('## '):
        name=line[3:]
        if name in ('References','Acknowledgment'):
            references=name=='References';p=paragraph(name,'Heading 5')
        else:
            p=paragraph(roman[section_num]+'.  '+re.sub(r'^\d+\s+','',name),'Heading 1');section_num+=1
        no_numbering(p);p.paragraph_format.keep_with_next=True;continue
    if line.startswith('### '):
        number,name=re.match(r'### \d+\.(\d+) (.*)',line).groups()
        p=paragraph(chr(64+int(number))+'. '+name,'Heading 2');no_numbering(p)
        p.paragraph_format.left_indent=Pt(0);p.paragraph_format.first_line_indent=Pt(0);continue
    if line.startswith('|'):
        block=[line]
        while i<len(lines) and lines[i].strip().startswith('|'):block.append(lines[i].strip());i+=1
        table(block);continue
    if re.match(r'^Table [IV]+  ',line):
        p=paragraph(line.replace('Table ','TABLE ').replace('  ','.  ',1),'table head');no_numbering(p);p.paragraph_format.keep_with_next=True;continue
    picture=re.fullmatch(r'!\[(.*?)\]\((.*?)\)',line)
    if picture:
        caption,path=picture.groups();file=ROOT/'paper'/path;equation=file.name.startswith('eq_')
        p=paragraph('','equation' if equation else 'Normal');p.paragraph_format.keep_with_next=not equation
        if equation:
            p.alignment=Align.LEFT;p.paragraph_format.tab_stops.clear_all()
            p.paragraph_format.tab_stops.add_tab_stop(Inches(WIDTH/2),Tab.CENTER)
            p.paragraph_format.tab_stops.add_tab_stop(Inches(WIDTH),Tab.RIGHT)
            p.add_run('\t');width=min(2.9,Image.open(file).width/240*2/3)
        else:width=WIDTH;p.alignment=Align.CENTER
        shape=p.add_run().add_picture(str(file),width=Inches(width));shape._inline.docPr.set('descr',caption)
        if equation:
            r=p.add_run('\t('+re.search(r'Equation (\d+)',caption).group(1)+')');r.font.name='Times New Roman';r.font.size=Pt(10)
            # Align the equation number to the optical center of the illustration.
            height=Image.open(file).height/Image.open(file).width*width*72
            position=OxmlElement('w:position');position.set(qn('w:val'),str(round(max(0,height/2-4)*2)))
            r._r.get_or_add_rPr().append(position)
        else:
            p=paragraph(re.sub(r'^(Fig\. \d+) ',r'\1. ',caption)+'.','figure caption');no_numbering(p)
        continue
    p=paragraph(line,'references' if references else 'Body Text')
    if references:
        no_numbering(p);p.paragraph_format.left_indent=Inches(.23);p.paragraph_format.first_line_indent=Inches(-.23)

# The official template ends with a continuous single-column section. Retain
# that mechanism to balance the last page's two text columns.
boundary=paragraph('','Normal')
boundary.paragraph_format.space_after=Pt(0);boundary.paragraph_format.line_spacing=Pt(1)
boundary.add_run().font.size=Pt(1)
boundary._p.get_or_add_pPr().append(deepcopy(body_section))
ending=deepcopy(body_section);ending.find(qn('w:cols')).set(qn('w:num'),'1')
doc.element.body.replace(doc.element.body.sectPr,ending)
OUT.parent.mkdir(parents=True,exist_ok=True);doc.save(OUT)
# Preserve the normalized design parts byte-for-byte.
with ZipFile(OUT) as z:package={e.filename:z.read(e.filename) for e in z.infolist()}
with ZipFile(template) as z:
    for name in ('word/styles.xml','word/numbering.xml','word/fontTable.xml','word/theme/theme1.xml'):
        if name in z.namelist():package[name]=z.read(name)
with ZipFile(OUT,'w',ZIP_DEFLATED) as z:
    for name,payload in package.items():z.writestr(name,payload)
print(OUT)
