import chainlit as cl
from dotenv import load_dotenv
from langchain_aws import ChatBedrockConverse, BedrockEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_postgres import PGVector
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document, StrOutputParser
from langchain.schema.runnable import Runnable, RunnableConfig
from sqlalchemy.ext.asyncio import create_async_engine
from typing import cast, Optional
import os
import psycopg2
import psycopg2.extras
import json


load_dotenv()

db_user = os.getenv("PGUSER")
db_password = os.getenv("PGPASSWORD")
db_host = os.getenv("PGHOST")
db_port = os.getenv("PGPORT")
db_name = os.getenv("PGDATABASE")
connection = (
    f"postgresql+psycopg://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
)


@cl.on_chat_start
async def on_chat_start():
    vectorstore = __get_vectorstore()
    retriever = __get_retriever(vectorstore)
    qa_prompt = __get_qa_prompt()
    llm = __get_llm()

    runnable_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | qa_prompt
        | llm
        | StrOutputParser()
    )
    # runnable_chain.get_graph().print_ascii()

    cl.user_session.set("runnable", runnable_chain)


@cl.on_message
async def on_message(message: cl.Message):
    runnable = cast(Runnable, cl.user_session.get("runnable"))

    msg = cl.Message(content="")

    async for chunk in runnable.astream(
        message.content,
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await msg.stream_token(chunk)

    await msg.send()


def __get_vectorstore(recreate_from: Optional[list[Document]] = None):
    embeddings = BedrockEmbeddings(
        model_id="amazon.titan-embed-text-v2:0",
        region_name="us-east-1",
    )

    if recreate_from is None:
        return PGVector(
            connection=create_async_engine(connection),
            embeddings=embeddings,
            use_jsonb=True,
        )
    else:
        return PGVector.from_documents(
            documents=recreate_from,
            embedding=embeddings,
            connection=connection,
            pre_delete_collection=True,  # 既存のコレクションを削除する場合
        )


def __get_retriever(vectorstore: PGVector):
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3, "include_metadata": True},
    )


def __get_qa_prompt():
    # [ORIGINAL]
    # """Human: You are a helpful assistant that answers questions directly and only using the information provided in the context below.
    # Guidance for answers:
    #     - Always use English as the language in your responses.
    #     - In your answers, always use a professional tone.
    #     - Begin your answers with "Based on the context provided: "
    #     - Simply answer the question clearly and with lots of detail using only the relevant details from the information below. If the context does not contain the answer, say "Sorry, I didn't understand that. Could you rephrase your question?"
    #     - Use bullet-points and provide as much detail as possible in your answer.
    #     - Always provide a summary at the end of your answer.
    #
    # Now read this context below and answer the question at the bottom.
    #
    # Context: {context}
    #
    # Question: {question}
    #
    # Assistant:"""
    system_prompt = """Human: You are a helpful assistant that answers questions directly and only using the information provided in the context below.
      Guidance for answers:
          - Always use Japanese as the language in your responses.
          - In your answers, always use a professional tone.
          - Begin your answers with "提供されたcontextに基づいて回答します。"
          - Simply answer the question clearly and with lots of detail using only the relevant details from the information below. If the context does not contain the answer, say "Sorry, I didn't understand that. Could you rephrase your question?"
          - Use bullet-points and provide as much detail as possible in your answer.
          - Always provide a summary at the end of your answer.

      Now read this context below and answer the question at the bottom.

      Context: {context}"""

    return ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{question}"),
        ]
    )


def __get_llm():
    return ChatBedrockConverse(
        # model="anthropic.claude-3-sonnet-20240229-v1:0",
        # model="anthropic.claude-3-5-sonnet-20241022-v2:0",
        model="amazon.nova-micro-v1:0",
        # model="amazon.nova-lite-v1:0",
        # model="amazon.nova-pro-v1:0",
        region_name="us-east-1",
        temperature=0.5,
    )


# ##############################################
# refill vectorstore
# ##############################################


def __fill_vectorstore():
    ids, docs = __get_movie_docs()

    vectorstore = __get_vectorstore(recreate_from=docs)

    print("Successfully!")


def __get_movie_docs():
    dbcur = __get_movies_as_json()

    text_splitter = RecursiveCharacterTextSplitter(
        separators=['",', "]", "}"],
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    ids = []
    docs = []
    for result in dbcur:
        movie_dict = result.get("json_build_object")
        print(movie_dict.get("id"))
        ids.append(movie_dict.get("id"))
        docs.append(
            Document(
                page_content=json.dumps(movie_dict),
                metadata=movie_dict,
            )
        )

    return ids, text_splitter.split_documents(docs)


def __get_movies_as_json():
    with psycopg2.connect(
        database=db_name,
        host=db_host,
        port=db_port,
        user=db_user,
        password=db_password,
    ) as dbconn:
        dbcur = dbconn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        dbcur.execute("""
select json_build_object(
    'id', m.id,
    'title', m.title,
    'overview', m.overview,
    'vote_average', m.vote_average,
    'vote_count', m.vote_count,
    'popularity', m.popularity,
    'credits', m.credits,
    'keywords', m.keywords,
    'genres', m.genre_id,
    'reviews', json_agg(json_build_object(
        'rating', r.rating,
        'review', r.review
    )) filter (where r.id is not null)
)
from movie.movies m
left outer join movie.reviews r ON m.id = r.id
where m.movie_embedding is null
group by m.id
limit 5
-- limit 500;
        """)

        return dbcur
