from typing import List, Dict
import re

from django.utils.lorem_ipsum import paragraphs
from langchain.text_splitter import RecursiveCharacterTextSplitter
import logging

logger = logging.getLogger(__name__)

class TextChunker:

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def chunk_text(self, text: str, metadata: Dict = None) -> List[Dict]:

        try:
            chunks = self.splitter.split_text(text)

            result = []
            for idx, chunk in enumerate(chunks):
                chunk_data = {
                    'content': chunk,
                    'chunk_index': idx,
                    'metadata': metadata or {},
                }
                result.append(chunk_data)

            logger.info(f"Created {len(result)} chunks from text of length {len(text)}")
            return result

        except Exception as e:
            logger.error(e)
            raise

    def chunk_by_sections(self, sections_dict: Dict[str, str]) -> List[Dict]:

        all_chunks = []

        for section_name, section_text in sections_dict.items():
            chunks = self.splitter.split_text(section_text)

            for idx, chunk in enumerate(chunks):
                chunk_data = {
                    'content': chunk,
                    'chunk_index': len(all_chunks),
                    'metadata': {
                        'section': section_name,
                        'section_chunk_index': idx,
                    }
                }
                all_chunks.append(chunk_data)

        return all_chunks
    def smart_chunk(self, text: str, page_mapping: Dict = None) -> List[Dict]:

        paragraphs = text.split("\n\n")

        chunks = []
        current_chunk = ""
        current_page = 1

        for para in paragraphs:

            if len(current_chunk) + len(para) > self.chunk_size:
                if current_chunk:
                    chunks.append({
                        'content': current_chunk.strip(),
                        'page_number': current_page,
                    })
                current_chunk = para
            else:
                current_chunk += "\n\n" + para

            if page_mapping:
                current_page = self._get_page_number(len(current_chunk), page_mapping)

        if current_chunk:
            chunks.append({
                'content': current_chunk.strip(),
                'page_number': current_page,
            })

        for idx, chunk in enumerate(chunks):
            chunk['chunk_index'] = idx

        return chunks

    def _get_page_number(self, char_position: int, page_mapping: Dict) -> int:

        for page_num, char_range in page_mapping.items():
            if char_range[0] <= char_position <= char_range[1]:
                return page_num
        return 1