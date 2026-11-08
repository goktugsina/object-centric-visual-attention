"""Prepare local submission metadata and inspect the existing PDF; no upload.

Requires pypdf in the document-authoring environment. Does not choose a license,
claim acceptance, modify the manuscript, or change repository visibility.
"""
import hashlib
import json
from pathlib import Path
from pypdf import PdfReader
from pypdf.generic import DictionaryObject, IndirectObject, ArrayObject

ROOT = Path(__file__).resolve().parents[1]


def inspect_pdf(path):
    reader = PdfReader(path)
    if reader.is_encrypted:
        raise ValueError('Submission PDF is encrypted')
    fonts = {}
    for page in reader.pages:
        for value in page['/Resources'].get('/Font', {}).values():
            font = value.get_object()
            for part in font.get('/DescendantFonts', [font]):
                part = part.get_object()
                descriptor = part.get('/FontDescriptor')
                descriptor = descriptor.get_object() if descriptor else {}
                name = str(part.get('/BaseFont', font.get('/BaseFont', 'unknown')))
                fonts[name] = {
                    'type': str(part.get('/Subtype')),
                    'embedded': any(key in descriptor for key in ('/FontFile', '/FontFile2', '/FontFile3')),
                }
    assert fonts and all(f['embedded'] and f['type'] != '/Type3' for f in fonts.values()), fonts
    seen = set()
    javascript = []

    def visit(obj):
        if isinstance(obj, IndirectObject):
            key = (obj.idnum, obj.generation)
            if key in seen:
                return
            seen.add(key)
            obj = obj.get_object()
        if isinstance(obj, DictionaryObject):
            if '/JS' in obj or '/JavaScript' in obj or obj.get('/S') == '/JavaScript':
                javascript.append(True)
            for value in obj.values():
                visit(value)
        elif isinstance(obj, (ArrayObject, list)):
            for value in obj:
                visit(value)

    visit(reader.trailer)
    assert not javascript, 'Embedded JavaScript found'
    text = '\n'.join(page.extract_text() or '' for page in reader.pages)
    assert len(reader.pages) >= 6 and len(text) > 2000
    return {
        'pages': len(reader.pages), 'encrypted': False,
        'embedded_javascript_found': False, 'fonts': fonts,
        'extracted_text_characters': len(text),
        'scope': 'Local technical checks only; not arXiv processing or moderation approval. Equations and charts are raster illustrations; body text is extractable.',
    }


def main():
    manuscript = (ROOT/'paper/manuscript.md').read_text(encoding='utf-8')
    title = manuscript.splitlines()[0].removeprefix('# ').strip()
    author_block = [line.strip() for line in manuscript.split('### Abstract')[0].splitlines()[1:] if line.strip()]
    author = author_block[0]
    abstract = manuscript.split('### Abstract', 1)[1].split('Keywords:', 1)[0].strip()
    keywords = manuscript.split('Keywords:', 1)[1].split('\n', 1)[0].strip()
    pdf = ROOT/'output/pdf/object_attention_paper.pdf'
    report = inspect_pdf(pdf)
    details_path = ROOT/'paper/submission_details.json'
    details = json.loads(details_path.read_text(encoding='utf-8')) if details_path.exists() else {}
    if details.get('author_name', author) != author:
        raise ValueError('Submission author name does not match manuscript; reconcile before building metadata')
    if author_block[1:] != ([details['affiliation']] if details.get('affiliation') else []):
        raise ValueError('Submission affiliation does not match manuscript')
    metadata = {
        'status': 'local_draft_not_submitted',
        'publication_route': details.get('publication_route'),
        'arxiv_account_status': details.get('arxiv_account_status'),
        'title': title, 'authors': [{'name': author, 'affiliation': details.get('affiliation'), 'orcid': details.get('orcid')}],
        'abstract': abstract, 'keywords': [word.strip() for word in keywords.split(',')],
        'candidate_arxiv_category': 'cs.CV',
        'suggested_comments': f"{report['pages']} pages, 1 figure, 3 tables. Exploratory OSIE and public EMOd study.",
        'journal_reference': None, 'publication_doi': None, 'license': details.get('license'),
        'pdf': pdf.relative_to(ROOT).as_posix(),
        'pdf_sha256': hashlib.sha256(pdf.read_bytes()).hexdigest(),
        'author_actions_before_submission': [
            'Review the manuscript, analysis and references and confirm author details.',
            'Check account and any endorsement requirements.',
            'Choose the distribution license.',
            'Confirm repository visibility and decide how readers can access the code.',
        ],
    }
    out = ROOT/'output/submission'
    out.mkdir(parents=True, exist_ok=True)
    for name, value in (('metadata.json', metadata), ('pdf_checks.json', report)):
        (out/name).write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n', encoding='utf-8', newline='\n')
    print(json.dumps({'draft': 'output/submission/metadata.json', 'pdf_pages': report['pages'],
                      'embedded_fonts': len(report['fonts']), 'abstract_words': len(abstract.split()), 'uploaded': False}))


if __name__ == '__main__':
    main()
