from typing import List, Dict, Optional
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from django.conf import settings
import logging

logger = logging.getLogger(__name__)

class QAService:

    def __init__(self):
        self.llm = Ollama(
            base_url=settings.ML_CONFIG['OLLAMA_BASE_URL'],
            model=settings.ML_CONFIG['OLLAMA_MODEL'],
            temperatue=0.1,
        )

        self.qa_prompt = PromptTemplate(
            input_variables=['context', 'question', 'chat_history'],
            template="""
            You are a helpful AI assistant specialized in analyzing research papers.
            Use the following context from the paper to answer the question.
            If you cannot find the answer in the context, say so honestly.
            Always cite which part of the paper you're referencing.
            
            Previous conversation:
            {chat_history}
            
            Context from paper:
            {context}
            
            Question: {question}
            
            Answer (be specific and cite sources):
            """
        )

        self.qa_chain = LLMChain(llm=self.llm, prompt=self.qa_prompt)

    def answer_question(self,
                        question: str,
                        retrieved_chunks: List[Dict],
                        chat_history: Optional[List[Dict]] = None) -> Dict:

        try:
            context = self._format_context(retrieved_chunks)

            history_text = self._format_chat_history(chat_history or [])

            response = self.qa_chain.invoke({
                'context': context,
                'question': question,
                'chat_history': history_text
            })

            answer = response['text'].strip()

            confidence = self._calculate_confidence(retrieved_chunks)

            return {
                'answer': answer,
                'sources': [
                    {
                        "chunk_id": chunk["id"],
                        "content": chunk["content"][:200] + "...",
                        "page": chunk['metadata'].get('page_number'),
                        "section": chunk['metadata'].get('section'),
                        "similarity_score": chunk['similarity_score'],
                    }
                    for chunk in retrieved_chunks
                ],
                'confidence': confidence,
                "tokens_used": {
                    "prompt": len(context) + len(question),
                    "completion": len(answer)
                }
            }

        except Exception as e:
            logger.error(f"Failed to answer question: {e}")
            raise

    def _format_context(self, chunks: List[Dict]) -> str:

        context_parts = []

        for idx, chunk in enumerate(chunks, 1):
            metadata = chunk['metadata']
            section = metadata.get('section','Unknown')
            page = metadata.get('page_number', 'Unknown')

            context_parts.append(
                f"[Source {idx} - {section}, Page {page}]:\n{chunk['content']}\n"

            )

        return "\n".join(context_parts)

    def _format_chat_history(self, history: List[Dict]) -> str:

        if not history:
            return "No previous conversation."

        history_parts = []

        for msg in history[-5:]:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            history_parts.append(f"{role.capitalize()}: {content}")

        return "\n".join(history_parts)

    def _calculate_confidence(self, chunks: List[Dict]) -> float:

        if not chunks:
            return 0.0

        top_scores = sorted(
            [chunk['similarity_score'] for chunk in chunks],
            reverse=True,
        )[:3]

        return sum(top_scores) / len(top_scores)

    def suggest_question(self, paper_summary: str) -> List[str]:

        try:
            prompt = PromptTemplate(
                input_variables=["summary"],
                template="""
                    Based on this research paper summary, suggest 5 interesting questions 
                    that a reader might want to ask about the paper.
    
                    Summary:
                    {summary}
    
                    Questions (one per line):
                    """
            )
            chain = LLMChain(llm=self.llm, prompt=prompt)
            response = chain.invoke({"summary": paper_summary})

            question = [
                q.strip("- ").strip() for q in response['text'].strip().split("\n")
                        if q.strip()
            ]

            return question[:5]

        except Exception as e:
            logger.error(f"Failed to suggest question: {e}")
            return [
                "What are the main contributions of this paper?",
                "What methodology was used?",
                "What are the key findings?",
                "What are the limitations?",
                "What future work is suggested?"
            ]