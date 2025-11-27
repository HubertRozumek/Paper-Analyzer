import PyMuPDF
import pdfplumber
from typing import Dict, List, Tuple
import re
import logging

logger = logging.getLogger(__name__)

class PDFProcessor:

    def __init__(self):
        self.section_patterns = {
            'abstract': r'(?i)abstract',
            'introduction': r'(?i)introduction',
            'methodology': r'(?i)(methodology|methods|materials)',
            'results': r'(?i)results',
            'discussion': r'(?i)discussion',
            'conclusion': r'(?i)conclusion',
            'references': r'(?i)references',
        }

    def extract_text(self, pdf_path: str) -> Dict:

        try:
            doc = PyMuPDF.open(pdf_path)

            result = {
                'full_text': '',
                'pages': [],
                'metadata': {},
                'num_pages': len(doc),
                'section': {},
            }

            result['metadata'] = self._extract_metadata(doc)

            for page_num, page in enumerate(doc, 1):
                text = page.get_text()
                result['pages'].append({
                    'page_number': page_num,
                    'text': text,
                })
                result['full_text'] += f"\n\n[Page {page_num}]\n{text}"

            result['section'] = self._identify_sections(result['full_text'])

            doc.close()
            return result

        except Exception as e:
            logger.error(f'Error extracting text from {pdf_path}: {e}')
            raise

    def extract_with_pdfplumber(self, pdf_path: str) -> Dict:
        """
        Alternative using pdfplumber
        """
        try:
            with pdfplumber.open(pdf_path) as pdf:
                result = {
                    'full_text': '',
                    'pages': [],
                    'tables': [],
                    'num_pages': len(pdf.pages),
                }

                for page_num, page in enumerate(pdf.pages, 1 ):
                    text = page.extract_text()
                    result['pages'].append({
                        'page_number': page_num,
                        'text': text,
                    })
                    result['full_text'] += f"\n\n[Page {page_num}]\n{text}"

                    tables = page.extract_tables()
                    if tables:
                        for table in tables:
                            result['tables'].append({
                                'page': page_num,
                                'data': table,
                            })

                return result

        except Exception as e:
            logger.error(f'Error extracting text from {pdf_path}: {e}')
            raise

    def _extract_metadata(self, doc) -> Dict:

        metadata = doc.metadata
        return {
            'title': metadata.get['title'],
            'author': metadata.get['author'],
            'subject': metadata.get['subject'],
            'keywords': metadata.get['keywords'],
            'creator': metadata.get['creator'],
            'producer': metadata.get['producer'],
            'creation_date': metadata.get['creation_date'],
        }

    def _identify_sections(self, text: str) -> Dict:

        sections = {}

        for section_name, pattern in self.section_patterns.items():
            matches = list(re.finditer(pattern, text, re.MULTILINE))
            if matches:
                sections[section_name] = matches[0].start()

        return sections

    def extract_section_text(self, full_text: str, sections: Dict, section_name: str) -> str:

        if section_name not in sections:
            return ""

        start = sections[section_name]

        next_sections = {k: v for k, v in sections.items() if v > start}
        if next_sections:
            end = min(next_sections.values())
            return full_text[start:end]
        else:
            return full_text[start:]

