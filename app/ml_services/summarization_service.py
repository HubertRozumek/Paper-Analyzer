from transformers import pipeline
from typing import Dict, Optional
import logging
from django.conf import settings

logger = logging.getLogger(__name__)

class SummarizationService:

    def __init__(self, model_name: str = "facebook/bart-large-cnn"):
        try:
            self.summarizer = pipeline("summarization",
                                       model=model_name,
                                       device=-1)
            logger.info(f"Summarization service is ready.")
        except Exception as e:
            logger.error(f"Failed to summarization service: {e}")
            self.summarizer = None

    def summarize(self,
                  text: str,
                  max_length: int = 500,
                  min_length: int = 100,
                  strategy: str = "balanced") -> str:

        if not self.summarizer:
            return self._fallback_summarize(text, max_length)

        try:
            max_input_length = 1024
            if len(text) > max_input_length:
                text = text[:max_input_length]

            summary = self.summarizer(
                text,
                max_length=max_length,
                min_length=min_length,
                do_sample=False,
            )[0]['summary_text']

            return summary

        except Exception as e:
            logger.error(f"Failed to summarization service: {e}")
            return self._fallback_summarize(text, max_length)

    def generate_multi_length_summaries(self, text: str) -> Dict[str, str]:

        return {
            'short': self.summarize(text, max_length=200, min_length=100),
            'medium': self.summarize(text, max_length=500, min_length=200),
            'long': self.summarize(text, max_length=1000, min_length=400),
        }

    def summarize_with_llm(self, text: str, length: str = 'medium') -> str:

        try:
            from langchain.community.llms import Ollama
            from langchain.prompts import PromptTemplate

            llm = Ollama(
                base_url = settings.ML_CONFIG['OLLAMA_BASE_URL'],
                model=settings.ML_CONFIG['OLLAMA_MODEL'],
            )

            length_instructions = {
                'short': '2-3 sentences (about 100-200 words)',
                'medium': '2-3 sentences (about 500-600 words)',
                'long': '2-3 sentences (about 1000-2000 words)'}

            prompt = PromptTemplate(
                input_variables=['text', 'length'],
                template="""
                Summarize the following research paper in {length}.
                Focus on: main contributions, methodology, key findings, and conclusions.
                
                Paper text:
                {text}
                
                Summary:
                """
            )

            chain = prompt | llm
            summary = chain.invoke({
                'text': text[:8000],
                'length': length_instructions[length],
            })

            return summary.strip()

        except Exception as e:
            logger.error(f"Error with LLM summarization: {e}")
            return self.summarize(text)


    def _fallback_summarize(self, text: str, max_length: int) -> str:

        sentences = text.split('. ')

        avg_sentence_length = len(text) / len(sentences) if sentences else 0
        num_sentences = int(max_length / avg_sentence_length) if avg_sentence_length > 0 else 3
        num_sentences = max(2, min(num_sentences, len(sentences)))

        if len(sentences) <= num_sentences:
            return '. '.join(sentences)

        summary_sentences = [sentences[0]]

        middle_indices = [len(sentences) // 2]
        summary_sentences.extend([sentences[i] for i in middle_indices])

        summary_sentences.append(sentences[-1])

        return '. '.join(summary_sentences) + "."

    def extract_key_insights(self, text: str) -> Dict:

        try:
            from langchain_community.llms import Ollama
            from langchain.prompts import PromptTemplate
            import json

            llm = Ollama(
                base_url = settings.ML_CONFIG['OLLAMA_BASE_URL'],
                model = settings.ML_CONFIG['OLLAMA_MODEL'],
            )

            prompt = PromptTemplate(
                input_variables=['text'],
                template="""
                Analyze this research paper and extract the following information in JSON format:
                
                {{
                    "key_findings": ["finding 1", "finding 2", ...],
                    "methodology": "brief description of methods used",
                    "conclusions": "main conclusions",
                    "limitations": ["limitation 1", "limitation 2", ...],
                    "future_work": "suggested future research directions"
                }}
                
                Paper text:
                {text}
                
                JSON output:
                """
            )

            chain = prompt | llm
            response = chain.invoke({'text': text[:8000]})

            insights = json.loads(response.strip())
            return insights

        except Exception as e:
            logger.error(f"Failed to extract key insights: {e}")
            return {
                "key_findings": [],
                "methodology": "",
                "conclusions": "",
                "limitations": [],
                "future_work": "",
            }