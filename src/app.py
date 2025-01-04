from dotenv import load_dotenv
from htmlTemplates import css
from langchain_aws import ChatBedrockConverse
from langchain_community.embeddings import BedrockEmbeddings
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableSerializable
from langchain_postgres import PGVector
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
import os
import psycopg2
import psycopg2.extras
import streamlit as st
import traceback
import json


# Streamlit components
def main():
    # Set the page configuration for the Streamlit application, including the page title and icon.
    st.set_page_config(
        page_title="Generative AI Q&A with Amazon Bedrock, Aurora PostgreSQL and pgvector",
        layout="wide",
        page_icon=":books::parrot:",
    )
    st.write(css, unsafe_allow_html=True)

    logo_url = "static/Powered-By_logo-stack_RGB_REV.png"
    st.sidebar.image(logo_url, width=150)

    # Check if the conversation and chat history are not present in the session state and initialize them to None.
    if "conversation" not in st.session_state:
        vectorstore = PGVector(
            connection=connection, embeddings=get_embeddings(), use_jsonb=True
        )
        st.session_state.vectorstore = vectorstore
        st.session_state.conversation = get_conversation_chain(vectorstore)

    # A header with the text appears at the top of the Streamlit application.
    st.header(
        "Generative AI Q&A with Amazon Bedrock, Aurora PostgreSQL and pgvector :books::parrot:"
    )

    # Create a text input box where you can ask questions about your documents.
    user_question = st.text_input(
        "Ask a question about your documents:", placeholder="What is Amazon Aurora?"
    )

    # Define a Go button for user action
    go_button = st.button("Submit", type="secondary")

    # If the go button is pressed or the user enters a question, it calls the handle_userinput() function to process the user's input.
    if go_button or user_question:
        with st.spinner("Processing..."):
            handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")

        if st.button("Process"):
            with st.spinner("Processing"):
                ids, docs = get_movie_docs()

                vectorstore = PGVector.from_documents(
                    documents=docs,
                    embedding=get_embeddings(),
                    # collection_name="movie.movies",
                    connection=connection,
                    pre_delete_collection=True,  # 既存のコレクションを削除する場合
                )

                st.write(json.dumps(ids))
                st.session_state.vectorstore = vectorstore
                st.session_state.conversation = get_conversation_chain(vectorstore)

                st.divider()

                st.success("Successfully!", icon="✅")

    with st.sidebar:
        st.divider()


def get_movie_docs():
    dbcur = get_movies_as_json()

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
        ids.append(movie_dict.get("id"))
        docs.append(
            Document(
                page_content=json.dumps(movie_dict),
                metadata=movie_dict,
            )
        )

    return ids, text_splitter.split_documents(docs)


# This function is responsible for processing the user's input question and generating a response from the chatbot
def handle_userinput(user_question):
    try:
        messages = st.session_state.conversation(user_question)

    except ValueError:
        st.write("Sorry, I didn't understand that. Could you rephrase your question?")
        print(traceback.format_exc())
        return

    for i, message in enumerate(messages):
        message.pretty_print()
        if isinstance(message, HumanMessage):
            st.write(message.content)
        else:
            st.success(message.content, icon="🤔")


def get_embeddings():
    return BedrockEmbeddings(
        model_id="amazon.titan-embed-text-v2:0",
        region_name="us-east-1",
    )


def get_conversation_chain(vectorstore: PGVector):
    llm = ChatBedrockConverse(
        # model="anthropic.claude-3-sonnet-20240229-v1:0",
        # model="anthropic.claude-3-5-sonnet-20241022-v2:0",
        model="amazon.nova-micro-v1:0",
        # model="amazon.nova-lite-v1:0",
        # model="amazon.nova-pro-v1:0",
        region_name="us-east-1",
        temperature=0.5,
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3, "include_metadata": True},
    )

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
    system_prompt = """Human: あなたは、以下に提供されたコンテキストのみを使用して、質問に直接かつ明確に答える、助けになるアシスタントです。
    回答のガイダンス:
        - 常に日本語で回答してください。
        - 回答では常にプロフェッショナルな口調を使用してください。
        - 回答は常に「提供されたcontextに基づいて回答します。」で始めてください。
        - 以下に示す情報から関連する詳細のみを使用して、質問に明確かつ詳細に答えてください。contextに答えが含まれていない場合は、「すみません、よく理解できませんでした。質問を言い換えてください。」と回答してください。
        - できるだけ多くの詳細を箇条書きで提供してください。
        - 常に回答の最後に要約を含めてください。

    次に、以下のコンテキストを読んで、一番下の質問に答えてください。

    Context: {context}"""

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("placeholder", "{chat_history}"),
            ("human", "{question}"),
        ]
    )

    runnable_chain: RunnableSerializable = (
        {"context": retriever, "question": RunnablePassthrough()} | qa_prompt | llm
    )

    def call_model(state: MessagesState):
        response = runnable_chain.invoke(state["messages"][-1].content)
        return {"messages": [response]}

    memory = MemorySaver()
    app = (
        StateGraph(MessagesState)
        .add_edge(START, "model")
        .add_node("model", call_model)
        .compile(checkpointer=memory)
    )

    def invoke(question: str):
        input_messages = [HumanMessage(question)]
        output = app.invoke(
            {"messages": input_messages},
            config={"configurable": {"thread_id": "abc123"}},
        )
        return output["messages"]

    return invoke


def get_movies_as_json():
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
;
        """)

        return dbcur


if __name__ == "__main__":
    load_dotenv()

    db_user = os.getenv("PGUSER")
    db_password = os.getenv("PGPASSWORD")
    db_host = os.getenv("PGHOST")
    db_port = os.getenv("PGPORT")
    db_name = os.getenv("PGDATABASE")
    connection = (
        f"postgresql+psycopg://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    )

    main()
