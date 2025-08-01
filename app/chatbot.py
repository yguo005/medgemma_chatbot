import os
import traceback
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.vectorstores import VectorStoreRetriever
from app.config import DB_FAISS_PATH, EMBEDDING_MODEL, LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS

class Chatbot:
    def __init__(self, openai_api_key: str):
        if not openai_api_key:
            raise ValueError("❌ ERROR: OpenAI API Key is missing!")
        self.openai_api_key = openai_api_key
        self.vectorstore = self._get_vectorstore()
        self.llm = self._load_llm()
        self.retrieval_chain = self._create_retrieval_chain()

    def _get_vectorstore(self):
        """Loads the FAISS vector store."""
        try:
            embedding_model = OpenAIEmbeddings(model=EMBEDDING_MODEL, openai_api_key=self.openai_api_key)
            vectorstore = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
            
            # Simple check for index dimension
            index = vectorstore.index
            test_vector = embedding_model.embed_query("Test")
            if index.d != len(test_vector):
                raise ValueError("FAISS index dimension mismatch.")

            return vectorstore
        except Exception as e:
            print(f"❌ ERROR: Failed to load FAISS vector store: {e}")
            return None

    def _load_llm(self):
        """Loads the language model."""
        return ChatOpenAI(
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
            openai_api_key=self.openai_api_key
        )

    def _create_retrieval_chain(self):
        """Creates the retrieval chain."""
        if not self.vectorstore:
            raise ValueError("Vector store not loaded.")

        retriever = VectorStoreRetriever(vectorstore=self.vectorstore)
        
        system_prompt = (
            "You are an expert in medical knowledge. "
            "Use the given context to answer the question. "
            "If you don't know the answer, say you don't know. "
            "Use three sentences maximum and keep the answer concise. "
            "Context: {context}"
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])

        return RunnableParallel(
            {"context": retriever, "input": RunnablePassthrough()}
        ) | prompt | self.llm

    def get_response(self, query: str):
        """Gets a response from the chatbot."""
        try:
            result = self.retrieval_chain.invoke(query)
            # Ensure the response is a string
            if hasattr(result, 'content'):
                return result.content
            return str(result)
        except Exception as e:
            print(f"❌ ERROR in get_response: {e}\n{traceback.format_exc()}")
            return "Sorry, I encountered an error." 