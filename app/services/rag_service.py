import os
import tempfile
import logging
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from app.models.emotion import emotion_info

logger = logging.getLogger(__name__)

# 전역 벡터 저장소
_vectorstore = None

def setup_rag_system():
    """RAG 시스템 초기화: 문서 로드, 분할, 임베딩"""
    try:
        docs = []
        with tempfile.TemporaryDirectory() as temp_dir:
            # 감정 정보를 임시 파일로 저장하고 로드
            for emotion, info in emotion_info.items():
                temp_file = os.path.join(temp_dir, f"{emotion}.txt")
                with open(temp_file, "w", encoding="utf-8") as f:
                    f.write(info)
                docs.extend(TextLoader(temp_file, encoding="utf-8").load())

            # 문서 분할 및 임베딩
            splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
            return FAISS.from_documents(
                splitter.split_documents(docs),
                HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask"),
            )
    except Exception as e:
        logger.error(f"RAG 시스템 초기화 오류: {str(e)}")
        return None

def get_vectorstore():
    """전역 벡터 저장소 반환 (초기화되지 않았다면 초기화)"""
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = setup_rag_system()
    return _vectorstore

def retrieve_relevant_context(query, k=3):
    """쿼리와 관련된 문서 검색"""
    try:
        vectorstore = get_vectorstore()
        if vectorstore is None:
            logger.error("벡터 저장소가 초기화되지 않았습니다")
            return ""
            
        documents = vectorstore.similarity_search(query, k=k)
        return "\n".join([doc.page_content for doc in documents])
    except Exception as e:
        logger.error(f"컨텍스트 검색 오류: {str(e)}")
        return ""