from __future__ import annotations

import streamlit as st
import requests
from pathlib import Path

from langchain.agents import initialize_agent, tool
from langchain.agents import AgentType
from langchain.agents import load_tools, initialize_agent, AgentType
from langchain.chat_models import AzureChatOpenAI
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

# Setup credentials in Streamlit
user_openai_api_key = st.sidebar.text_input(
    "OpenAI API Key", type="password", help="Set this to run your own custom questions."
)

# # Add a radio button selector for the model
# model_name = st.sidebar.radio(
#     "Select Model",
#     ("gpt-3.5-turbo", "gpt-4"),
#     help="Select the model to use for the chat.",
# )

endpoint = "http://localhost:8080/query"


def get_schema():
    """Query the api for its schema"""
    global endpoint
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
    request = requests.post(endpoint, json={"query": query})
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

    global endpoint
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
    IsDependency(isDependencySpec: { package: { type: "guac", name: "alpine" }}) {
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

    ## Use this query when user asks what images depend on a package (like logrus) Do not use namespace for each image.
    query IsDependencyQ2 {
    IsDependency(isDependencySpec: {
        package: { type: "guac" }
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

    Answer the following question: {query} by using either terminal or the graphql database that has this schema {graphql_fields}. You MUST NOT use ``` to start your query."""

    try:
        result = agent.run(prompt)
    except Exception as e:
        prompt += f"\n\nThere was an error with the request.\nError: {e}\n\nPlease reformat GraphQL query (avoid issues with backticks if possible)."
        result = agent.run(prompt)

    return result


tools = []

if user_openai_api_key:
    openai_api_key = user_openai_api_key
    enable_custom = True
    llm = AzureChatOpenAI(
        openai_api_key=openai_api_key,
        openai_api_version="2023-07-01-preview",
        deployment_name="gpt-4-32k-0613",
        openai_api_type="azure",
        temperature=0,
        streaming=True,
    )

else:
    openai_api_key = "not_supplied"
    enable_custom = False
    llm = AzureChatOpenAI(
        openai_api_key=openai_api_key,
        openai_api_version="2023-07-01-preview",
        deployment_name="gpt-4-32k-0613",
        openai_api_type="azure",
        temperature=0,
        streaming=True,
    )

tools = load_tools(
    ["graphql", "terminal"],
    graphql_endpoint=endpoint,
    llm=llm,
)

# Initialize agent
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)

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

    # st.write(f"Checking if {path_user_input} is in {SAVED_SESSIONS.keys()}")

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
