import asyncio
from typing import Any
from langchain_community.vectorstores import FAISS
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.documents import Document
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from pydantic import SecretStr
from task._constants import DIAL_URL, API_KEY
from task.user_client import UserClient

#TODO:
# Before implementation open the `vector_based_grounding.png` to see the flow of app

#TODO:
# Provide System prompt. Goal is to explain LLM that in the user message will be provide rag context that is retrieved
# based on user question and user question and LLM need to answer to user based on provided context
SYSTEM_PROMPT = """
"""

#TODO:
# Should consist retrieved context and user question
USER_PROMPT = """
"""


def format_user_document(user: dict[str, Any]) -> str:
    #TODO:
    # Prepare context from users JSONs in the same way as in `no_grounding.py` `join_context` method (collect as one string)
    raise NotImplementedError


class UserRAG:
    def __init__(self, embeddings: AzureOpenAIEmbeddings, llm_client: AzureChatOpenAI):
        self.llm_client = llm_client
        self.embeddings = embeddings
        self.vectorstore = None

    async def __aenter__(self):
        print("🔎 Loading all users...")
        user_client = UserClient()
        user_client.health()
        users = UserClient.get_all_users()

        print(f"Formatting {len(users)} user documents...")
        documents = [
            Document(
                page_content=format_user_document(user)
            )
            for user in users
        ]

        print(f"↗️ Creating embeddings and vectorstore for {len(documents)} documents...")
        self.vectorstore = await self._create_vectorstore_with_batching(documents, batch_size=100)
        print("✅ Vectorstore is ready.")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    async def _create_vectorstore_with_batching(self, documents: list[Document], batch_size: int = 100):
        batches = [documents[i:i + batch_size] for i in range(0, len(documents), batch_size)]

        batch_tasks = [FAISS.afrom_documents(batch, self.embeddings) for batch in batches]
        batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
        
        final_vectorstore = None
        for batch_vectorstore in batch_results:
            if batch_vectorstore is not None:
                if final_vectorstore is None:
                    final_vectorstore = batch_vectorstore
                else:
                    final_vectorstore.merge_from(batch_vectorstore)

        if final_vectorstore is None:
            raise Exception("All batches failed to process")

        return final_vectorstore

    async def retrieve_context(self, query: str, k: int = 10, score: float = 0.1) -> str:
        print("Retrieving context...")
        relevant_docs = self.vectorstore.similarity_search_with_relevance_scores(
            query, k=k, score_threshold=score
        )

        context_parts = []
        for doc, relevance_score in relevant_docs:
            context_parts.append(doc.page_content)
            print(f"Retrieved (Score: {relevance_score:.3f}): {doc.page_content}")
        print(f"{'=' * 100}\n")

        return "\n\n".join(context_parts)
    
    def augment_prompt(self, query: str, context: str) -> str:
        return USER_PROMPT.format(context=context, query=query)

    def generate_answer(self, augmented_prompt: str) -> str:
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=augmented_prompt)
        ]
        response = self.llm_client.invoke(messages)
        return response.content



async def main():
    embeddings = AzureOpenAIEmbeddings(
        deployment='text-embedding-3-small-1',
        azure_endpoint=DIAL_URL,
        api_key=SecretStr(API_KEY),
        dimensions=384,
    )

    llm_client = AzureChatOpenAI(
        temperature=0.0,
        azure_deployment='gpt-4o',
        azure_endpoint=DIAL_URL,
        api_key=SecretStr(API_KEY),
        api_version="",
    )

    try:
        async with UserRAG(embeddings, llm_client) as rag:
            print("Query samples:")
            print(" - I need user emails that filled with hiking and psychology")
            print(" - Who is John?")
            while True:
                user_question = input("> ").strip()
                if user_question.lower() in ['quit', 'exit']:
                    break

                try:
                    context = await rag.retrieve_context(user_question)
                    augmented_prompt = rag.augment_prompt(user_question, context)
                    answer = rag.generate_answer(augmented_prompt)
                    print(answer)
                except Exception as e:
                    print(f"❌ Error processing question: {e}")

    except Exception as e:
        print(f"❌ Failed to initialize RAG system: {e}")


asyncio.run(main())
