from __future__ import annotations

import streamlit as st
import requests
import os
from pathlib import Path

from langchain.agents import initialize_agent, tool
from langchain.agents import AgentType
from langchain.agents import load_tools, initialize_agent, AgentType
from langchain.chat_models import AzureChatOpenAI, ChatOpenAI
from langchain.schema.output_parser import OutputParserException
from langchain.utilities import GraphQLAPIWrapper

from callbacks.capturing_callback_handler import CapturingCallbackHandler, playback_callbacks
from callbacks.streamlit_callback_handler import StreamlitCallbackHandler

from utils.clear_results import with_clear_container

st.set_page_config(
    page_title="Guac-AI-Mole",
    page_icon="🥑",
    initial_sidebar_state="collapsed",
)

runs_dir = Path(__file__).parent / "runs"
runs_dir.mkdir(exist_ok=True)

SAVED_SESSIONS = {}
# Populate saved sessions from runs_dir
for path in runs_dir.glob("*.pickle"):
    with open(path, "rb") as f:
        SAVED_SESSIONS[path.stem] = path


"# 🥑 Guac-AI-Mole"
"Charting the Course for Secure Software Supply Chain"
"Ask questions about your software supply chain and get answers from the Guac-AI-Mole!"

openai_api_key = os.getenv("OPENAI_API_KEY")
user_openai_api_key = st.sidebar.text_input(
    "OpenAI API Key", type="password", help="Set this to your own OpenAI API key.", value=openai_api_key
)

openai_api_endpoint = os.getenv("OPENAI_API_ENDPOINT")
user_openai_api_endpoint = st.sidebar.text_input(
    "OpenAI API Endpoint", type="default", help="Set this to your own OpenAI endpoint.", value=openai_api_endpoint
)

openai_api_model = os.getenv("OPENAI_API_MODEL")
user_openai_model = st.sidebar.text_input(
    "OpenAI Model", type="default", help="Set this to your own OpenAI model or deployment name.", value=openai_api_model
)

graphql_endpoint = os.getenv("GUAC_GRAPHQL_ENDPOINT")
user_graphql_endpoint = st.sidebar.text_input(
    "GUAC GraphQL Endpoint", type="default", help="Set this to your own GUAC GraphQL endpoint.", value=graphql_endpoint
)

def get_schema():
    """Query the api for its schema"""
    global user_graphql_endpoint
    query = """
    query IntrospectionQuery {
        __schema {
            types {
                name
                kind
                fields {
                    name
                    type {
                        name
                        kind
                        ofType {
                            name
                            kind
                        }
                    }
                }
            }
        }
    }"""
    request = requests.post(user_graphql_endpoint, json={"query": query})
    json_output = request.json()

    # Simplify the schema
    simplified_schema = {}
    for type_info in json_output["data"]["__schema"]["types"]:
        if not type_info["name"].startswith("__"):
            fields = type_info.get("fields")
            if fields is not None and fields is not []:
                simplified_schema[type_info["name"]] = {
                    "kind": type_info["kind"],
                    "fields": ", ".join(
                        [
                            field["name"]
                            for field in fields
                            if not field["name"].startswith("__")
                        ]
                    ),
                }
            else:
                simplified_schema[type_info["name"]] = {
                    "kind": type_info["kind"],
                }

    return simplified_schema


@tool
def answer_question(query: str):
    """Answer a question using graphql API"""

    global user_graphql_endpoint
    graphql_fields = (
        get_schema()
    )
    image_example = """
    ## List running images using terminal tool
    kubectl get pods --all-namespaces -o go-template --template='{{range .items}}{{range .spec.containers}}{{.image}} {{end}}{{end}}'
    """

    gql_examples = """
    ## Use this query when user asks what are dependencies of an image
    query IsDependencyQ1 {
    IsDependency(isDependencySpec: { package: { name: "alpine" }}) {
      dependentPackage {
      type
        namespaces {
          namespace
            names {
              name
            }
          }
        }
      }
    }

    ## Use this query when user asks what images depend on a package (like logrus).
    query IsDependencyQ2 {
    IsDependency(isDependencySpec: {
        package: { }
        dependentPackage: { name: "logrus" }
    }) {
      package {
        namespaces {
            namespace
            names {
              name
            }
          }
        }
      }
    }

    ## Use this query when user asks about a vulnerability id, this will return a package that has the vulnerability. You must query further with IsDependencyQ2 to see what images includes this package.
    query CertifyVulnQ1 {
    CertifyVuln(certifyVulnSpec: {vulnerability: {vulnerabilityID: "dsa-5122-1"}}) {
      package {
        namespaces {
            names {
              name
            }
          }
        }
      }
    }
    """

    prompt = f"""
    Do NOT, under any circumstances, use ``` anywhere.

    To check if an image is running, use the terminal tool to list all running images with kubectl. Example:
    {image_example} Only execute this based on the graphql answer, determine if the image is running.

    Consider the syntax as image name followed by a dash and tag. For example, if 'bar-latest' is returned as part of graphql query, and terminal output contains 'foo/bar:latest' then consider it as running.

    Here are some example queries for the graphql endpoint described below:
    {gql_examples}

    Answer the following question: {query} by using either terminal or the graphql database that has this schema {graphql_fields}. action_input should not contain a seperate query key. action_input should only have the query itself."""

    try:
        result = agent.run(prompt)
    except Exception as e:
        prompt += f"\n\nThere was an error with the request.\nError: {e}\n\nPlease reformat GraphQL query (avoid issues with backticks if possible)."
        result = agent.run(prompt)

    return result


tools = []
llm = None

if user_openai_api_key:
    enable_custom = True

    if user_openai_api_endpoint.endswith("azure.com"):
        print("Using Azure LLM")
        llm = AzureChatOpenAI(
            openai_api_key=user_openai_api_key,
            openai_api_base=user_openai_api_endpoint,
            openai_api_version="2023-08-01-preview",
            openai_api_type="azure",
            deployment_name=user_openai_model,
            temperature=0,
            streaming=True,
        )
    else:
        print("Using OpenAI or LocalAI LLM")
        llm = ChatOpenAI(
            openai_api_key=user_openai_api_key,
            openai_api_base=user_openai_api_endpoint,
            model_name=user_openai_model,
            temperature=0,
            streaming=True,
        )

    tools = load_tools(
        ["graphql", "terminal"],
        graphql_endpoint=user_graphql_endpoint,
        llm=llm,
    )

    # Initialize agent
    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )
else:
    enable_custom = False


with st.form(key="form"):
    if not enable_custom:
        "Ask one of the sample questions, or enter your API Key in the sidebar to ask your own custom questions."
    prefilled = (
        st.selectbox(
            "Sample questions",
            sorted([key.replace("_", " ") for key in SAVED_SESSIONS.keys()]),
        )
        or ""
    )
    user_input = ""

    if enable_custom:
        user_input = st.text_input("Or, ask your own question")
    if not user_input:
        user_input = prefilled
    submit_clicked = st.form_submit_button("Submit Question")

output_container = st.empty()

if with_clear_container(submit_clicked):
    output_container = output_container.container()
    output_container.chat_message("user").write(user_input)

    answer_container = output_container.chat_message("assistant", avatar="🥑")
    st_callback = StreamlitCallbackHandler(answer_container)

    # If we've saved this question, play it back instead of actually running LangChain
    # (so that we don't exhaust our API calls unnecessarily)
    path_user_input = "_".join(user_input.split(" "))

    if path_user_input in SAVED_SESSIONS.keys():
        print(f"Playing saved session: {user_input}")
        session_name = SAVED_SESSIONS[path_user_input]
        session_path = Path(__file__).parent / "runs" / session_name
        print(f"Playing saved session: {session_path}")
        answer = playback_callbacks(
            [st_callback], str(session_path), max_pause_time=1)
    else:
        print(f"Running LangChain: {user_input} because not in SAVED_SESSIONS")
        capturing_callback = CapturingCallbackHandler()
        try:
            answer = answer_question(user_input, callbacks=[
                                     st_callback, capturing_callback])
        except OutputParserException as e:
            answer = e.args[0].split("LLM output: ")[1]
        pickle_filename = user_input.replace(" ", "_") + ".pickle"
        capturing_callback.dump_records_to_file(runs_dir / pickle_filename)

    answer_container.write(answer)
