from __future__ import annotations
from langchain.prompts import PromptTemplate
from streamlit.delta_generator import DeltaGenerator
from typing import Any, TypedDict
import time
import pickle
from langchain.chains import LLMChain
from langchain.agents import initialize_agent, Tool, tool
from langchain.agents import AgentType
from pathlib import Path
import streamlit as st
import pandas as pd
import requests
from datetime import datetime
from langchain.chat_models import AzureChatOpenAI
from langchain.utilities import GraphQLAPIWrapper
from langchain.agents import load_tools, initialize_agent, AgentType
from langchain.schema.output_parser import OutputParserException


from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, NamedTuple, Optional, Union

from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.streamlit.mutable_expander import MutableExpander
from langchain.schema import AgentAction, AgentFinish, LLMResult
from langchain.callbacks import StreamlitCallbackHandler

st.set_page_config(
    page_title="Guac-AI-Mole",
    page_icon="🥑",
    initial_sidebar_state="collapsed",
)

runs_dir = Path(__file__).parent / "runs"
runs_dir.mkdir(exist_ok=True)

# This is intentionally not an enum so that we avoid serializing a
# custom class with pickle.
class CallbackType:
    ON_LLM_START = "on_llm_start"
    ON_LLM_NEW_TOKEN = "on_llm_new_token"
    ON_LLM_END = "on_llm_end"
    ON_LLM_ERROR = "on_llm_error"
    ON_TOOL_START = "on_tool_start"
    ON_TOOL_END = "on_tool_end"
    ON_TOOL_ERROR = "on_tool_error"
    ON_TEXT = "on_text"
    ON_CHAIN_START = "on_chain_start"
    ON_CHAIN_END = "on_chain_end"
    ON_CHAIN_ERROR = "on_chain_error"
    ON_AGENT_ACTION = "on_agent_action"
    ON_AGENT_FINISH = "on_agent_finish"


# We use TypedDict, rather than NamedTuple, so that we avoid serializing a
# custom class with pickle. All of this class's members should be basic Python types.
class CallbackRecord(TypedDict):
    callback_type: str
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    time_delta: float  # Number of seconds between this record and the previous one


def load_records_from_file(path: str) -> list[CallbackRecord]:
    """Load the list of CallbackRecords from a pickle file at the given path."""
    with open(path, "rb") as file:
        records = pickle.load(file)

    if not isinstance(records, list):
        raise RuntimeError(f"Bad CallbackRecord data in {path}")
    return records


def playback_callbacks(
    handlers: list[BaseCallbackHandler],
    records_or_filename: list[CallbackRecord] | str,
    max_pause_time: float,
) -> str:
    if isinstance(records_or_filename, list):
        records = records_or_filename
    else:
        records = load_records_from_file(records_or_filename)

    for record in records:
        pause_time = min(record["time_delta"] / 2, max_pause_time)
        if pause_time > 0:
            time.sleep(pause_time)

        for handler in handlers:
            if record["callback_type"] == CallbackType.ON_LLM_START:
                handler.on_llm_start(*record["args"], **record["kwargs"])
            elif record["callback_type"] == CallbackType.ON_LLM_NEW_TOKEN:
                handler.on_llm_new_token(*record["args"], **record["kwargs"])
            elif record["callback_type"] == CallbackType.ON_LLM_END:
                handler.on_llm_end(*record["args"], **record["kwargs"])
            elif record["callback_type"] == CallbackType.ON_LLM_ERROR:
                handler.on_llm_error(*record["args"], **record["kwargs"])
            elif record["callback_type"] == CallbackType.ON_TOOL_START:
                handler.on_tool_start(*record["args"], **record["kwargs"])
            elif record["callback_type"] == CallbackType.ON_TOOL_END:
                handler.on_tool_end(*record["args"], **record["kwargs"])
            elif record["callback_type"] == CallbackType.ON_TOOL_ERROR:
                handler.on_tool_error(*record["args"], **record["kwargs"])
            elif record["callback_type"] == CallbackType.ON_TEXT:
                handler.on_text(*record["args"], **record["kwargs"])
            elif record["callback_type"] == CallbackType.ON_CHAIN_START:
                handler.on_chain_start(*record["args"], **record["kwargs"])
            elif record["callback_type"] == CallbackType.ON_CHAIN_END:
                handler.on_chain_end(*record["args"], **record["kwargs"])
            elif record["callback_type"] == CallbackType.ON_CHAIN_ERROR:
                handler.on_chain_error(*record["args"], **record["kwargs"])
            elif record["callback_type"] == CallbackType.ON_AGENT_ACTION:
                handler.on_agent_action(*record["args"], **record["kwargs"])
            elif record["callback_type"] == CallbackType.ON_AGENT_FINISH:
                handler.on_agent_finish(*record["args"], **record["kwargs"])

    # Return the agent's result
    for record in records:
        if record["callback_type"] == CallbackType.ON_AGENT_FINISH:
            return record["args"][0][0]["output"]

    return "[Missing Agent Result]"


class CapturingCallbackHandler(BaseCallbackHandler):
    def __init__(self) -> None:
        self._records: list[CallbackRecord] = []
        self._last_time: float | None = None

    def dump_records_to_file(self, path: str) -> None:
        """Write the list of CallbackRecords to a pickle file at the given path."""
        with open(path, "wb") as file:
            pickle.dump(self._records, file)

    def _append_record(
        self, type: str, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> None:
        time_now = time.time()
        time_delta = time_now - self._last_time if self._last_time is not None else 0
        self._last_time = time_now
        self._records.append(
            CallbackRecord(
                callback_type=type, args=args, kwargs=kwargs, time_delta=time_delta
            )
        )

    def on_llm_start(self, *args: Any, **kwargs: Any) -> None:
        self._append_record(CallbackType.ON_LLM_START, args, kwargs)

    def on_llm_new_token(self, *args: Any, **kwargs: Any) -> None:
        self._append_record(CallbackType.ON_LLM_NEW_TOKEN, args, kwargs)

    def on_llm_end(self, *args: Any, **kwargs: Any) -> None:
        self._append_record(CallbackType.ON_LLM_END, args, kwargs)

    def on_llm_error(self, *args: Any, **kwargs: Any) -> None:
        self._append_record(CallbackType.ON_LLM_ERROR, args, kwargs)

    def on_tool_start(self, *args: Any, **kwargs: Any) -> None:
        self._append_record(CallbackType.ON_TOOL_START, args, kwargs)

    def on_tool_end(self, *args: Any, **kwargs: Any) -> None:
        self._append_record(CallbackType.ON_TOOL_END, args, kwargs)

    def on_tool_error(self, *args: Any, **kwargs: Any) -> None:
        self._append_record(CallbackType.ON_TOOL_ERROR, args, kwargs)

    def on_text(self, *args: Any, **kwargs: Any) -> None:
        self._append_record(CallbackType.ON_TEXT, args, kwargs)

    def on_chain_start(self, *args: Any, **kwargs: Any) -> None:
        self._append_record(CallbackType.ON_CHAIN_START, args, kwargs)

    def on_chain_end(self, *args: Any, **kwargs: Any) -> None:
        self._append_record(CallbackType.ON_CHAIN_END, args, kwargs)

    def on_chain_error(self, *args: Any, **kwargs: Any) -> None:
        self._append_record(CallbackType.ON_CHAIN_ERROR, args, kwargs)

    def on_agent_action(self, *args: Any, **kwargs: Any) -> Any:
        self._append_record(CallbackType.ON_AGENT_ACTION, args, kwargs)

    def on_agent_finish(self, *args: Any, **kwargs: Any) -> None:
        self._append_record(CallbackType.ON_AGENT_FINISH, args, kwargs)


# A hack to "clear" the previous result when submitting a new prompt. This avoids
# the "previous run's text is grayed-out but visible during rerun" Streamlit behavior.
class DirtyState:
    NOT_DIRTY = "NOT_DIRTY"
    DIRTY = "DIRTY"
    UNHANDLED_SUBMIT = "UNHANDLED_SUBMIT"


def get_dirty_state() -> str:
    return st.session_state.get("dirty_state", DirtyState.NOT_DIRTY)


def set_dirty_state(state: str) -> None:
    st.session_state["dirty_state"] = state


def with_clear_container(submit_clicked: bool) -> bool:
    if get_dirty_state() == DirtyState.DIRTY:
        if submit_clicked:
            set_dirty_state(DirtyState.UNHANDLED_SUBMIT)
            st.experimental_rerun()
        else:
            set_dirty_state(DirtyState.NOT_DIRTY)

    if submit_clicked or get_dirty_state() == DirtyState.UNHANDLED_SUBMIT:
        set_dirty_state(DirtyState.DIRTY)
        return True

    return False


SAVED_SESSIONS = {}
# Populate saved sessions from runs_dir
for path in runs_dir.glob("*.pickle"):
    with open(path, "rb") as f:
        SAVED_SESSIONS[path.stem] = path


"🥑 **Guac-AI-Mole**: Charting the Course for Secure Software Supply Chain"
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


#################################
# Customized Streamlit Callback #
#################################

def _convert_newlines(text: str) -> str:
    """Convert newline characters to markdown newline sequences
    (space, space, newline).
    """
    return text.replace("\n", "  \n")


CHECKMARK_EMOJI = "✅"
THINKING_EMOJI = ":thinking_face:"
HISTORY_EMOJI = ":books:"
EXCEPTION_EMOJI = "⚠️"


class LLMThoughtState(Enum):
    # The LLM is thinking about what to do next. We don't know which tool we'll run.
    THINKING = "THINKING"
    # The LLM has decided to run a tool. We don't have results from the tool yet.
    RUNNING_TOOL = "RUNNING_TOOL"
    # We have results from the tool.
    COMPLETE = "COMPLETE"


class ToolRecord(NamedTuple):
    name: str
    input_str: str


class LLMThoughtLabeler:
    """
    Generates markdown labels for LLMThought containers. Pass a custom
    subclass of this to StreamlitCallbackHandler to override its default
    labeling logic.
    """

    def get_initial_label(self) -> str:
        """Return the markdown label for a new LLMThought that doesn't have
        an associated tool yet.
        """
        return f"{THINKING_EMOJI} **Thinking...**"

    def get_tool_label(self, tool: ToolRecord, is_complete: bool) -> str:
        """Return the label for an LLMThought that has an associated
        tool.

        Parameters
        ----------
        tool
            The tool's ToolRecord

        is_complete
            True if the thought is complete; False if the thought
            is still receiving input.

        Returns
        -------
        The markdown label for the thought's container.

        """
        input = tool.input_str
        name = tool.name
        emoji = CHECKMARK_EMOJI if is_complete else THINKING_EMOJI
        if name == "_Exception":
            emoji = EXCEPTION_EMOJI
            name = "Parsing error"
        idx = min([60, len(input)])
        input = input[0:idx]
        if len(tool.input_str) > idx:
            input = input + "..."
        input = input.replace("\n", " ")
        label = f"{emoji} **{name}:** {input}"
        return label

    def get_history_label(self) -> str:
        """Return a markdown label for the special 'history' container
        that contains overflow thoughts.
        """
        return f"{HISTORY_EMOJI} **History**"

    def get_final_agent_thought_label(self) -> str:
        """Return the markdown label for the agent's final thought -
        the "Now I have the answer" thought, that doesn't involve
        a tool.
        """
        return f"{CHECKMARK_EMOJI} **Complete!**"


class LLMThought:
    def __init__(
        self,
        parent_container: DeltaGenerator,
        labeler: LLMThoughtLabeler,
        expanded: bool,
        collapse_on_complete: bool,
    ):
        self._container = MutableExpander(
            parent_container=parent_container,
            label=labeler.get_initial_label(),
            expanded=expanded,
        )
        self._state = LLMThoughtState.THINKING
        self._llm_token_stream = ""
        self._llm_token_writer_idx: Optional[int] = None
        self._last_tool: Optional[ToolRecord] = None
        self._collapse_on_complete = collapse_on_complete
        self._labeler = labeler

    @property
    def container(self) -> MutableExpander:
        """The container we're writing into."""
        return self._container

    @property
    def last_tool(self) -> Optional[ToolRecord]:
        """The last tool executed by this thought"""
        return self._last_tool

    def _reset_llm_token_stream(self) -> None:
        self._llm_token_stream = ""
        self._llm_token_writer_idx = None

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str]) -> None:
        self._reset_llm_token_stream()

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        # This is only called when the LLM is initialized with `streaming=True`
        self._llm_token_stream += _convert_newlines(token)
        self._llm_token_writer_idx = self._container.markdown(
            self._llm_token_stream, index=self._llm_token_writer_idx
        )

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        # `response` is the concatenation of all the tokens received by the LLM.
        # If we're receiving streaming tokens from `on_llm_new_token`, this response
        # data is redundant
        self._reset_llm_token_stream()

    def on_llm_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        self._container.markdown("**LLM encountered an error...**")
        self._container.exception(error)

    def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
    ) -> None:
        # Called with the name of the tool we're about to run (in `serialized[name]`),
        # and its input. We change our container's label to be the tool name.
        self._state = LLMThoughtState.RUNNING_TOOL
        tool_name = serialized["name"]
        self._last_tool = ToolRecord(name=tool_name, input_str=input_str)
        self._container.update(
            new_label=self._labeler.get_tool_label(
                self._last_tool, is_complete=False)
        )

    def on_tool_end(
        self,
        output: str,
        color: Optional[str] = None,
        observation_prefix: Optional[str] = None,
        llm_prefix: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        self._container.markdown(f"**{output}**")

    def on_tool_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        self._container.markdown("**Tool encountered an error...**")
        self._container.exception(error)

    def on_agent_action(
        self, action: AgentAction, color: Optional[str] = None, **kwargs: Any
    ) -> Any:
        # Called when we're about to kick off a new tool. The `action` data
        # tells us the tool we're about to use, and the input we'll give it.
        # We don't output anything here, because we'll receive this same data
        # when `on_tool_start` is called immediately after.
        pass

    def complete(self, final_label: Optional[str] = None) -> None:
        """Finish the thought."""
        if final_label is None and self._state == LLMThoughtState.RUNNING_TOOL:
            assert (
                self._last_tool is not None
            ), "_last_tool should never be null when _state == RUNNING_TOOL"
            final_label = self._labeler.get_tool_label(
                self._last_tool, is_complete=True
            )
        self._state = LLMThoughtState.COMPLETE
        if self._collapse_on_complete:
            self._container.update(new_label=final_label, new_expanded=False)
        else:
            self._container.update(new_label=final_label)

    def clear(self) -> None:
        """Remove the thought from the screen. A cleared thought can't be reused."""
        self._container.clear()


class StreamlitCallbackHandler(BaseCallbackHandler):
    def __init__(
        self,
        parent_container: DeltaGenerator,
        *,
        max_thought_containers: int = 4,
        expand_new_thoughts: bool = True,
        collapse_completed_thoughts: bool = True,
        thought_labeler: Optional[LLMThoughtLabeler] = None,
    ):
        """Create a StreamlitCallbackHandler instance.

        Parameters
        ----------
        parent_container
            The `st.container` that will contain all the Streamlit elements that the
            Handler creates.
        max_thought_containers
            The max number of completed LLM thought containers to show at once. When
            this threshold is reached, a new thought will cause the oldest thoughts to
            be collapsed into a "History" expander. Defaults to 4.
        expand_new_thoughts
            Each LLM "thought" gets its own `st.expander`. This param controls whether
            that expander is expanded by default. Defaults to True.
        collapse_completed_thoughts
            If True, LLM thought expanders will be collapsed when completed.
            Defaults to True.
        thought_labeler
            An optional custom LLMThoughtLabeler instance. If unspecified, the handler
            will use the default thought labeling logic. Defaults to None.
        """
        self._parent_container = parent_container
        self._history_parent = parent_container.container()
        self._history_container: Optional[MutableExpander] = None
        self._current_thought: Optional[LLMThought] = None
        self._completed_thoughts: List[LLMThought] = []
        self._max_thought_containers = max(max_thought_containers, 1)
        self._expand_new_thoughts = expand_new_thoughts
        self._collapse_completed_thoughts = collapse_completed_thoughts
        self._thought_labeler = thought_labeler or LLMThoughtLabeler()

    def _require_current_thought(self) -> LLMThought:
        """Return our current LLMThought. Raise an error if we have no current
        thought.
        """

        if self._current_thought is None:
            # print(
            #     "Current LLMThought is unexpectedly None!",
            #     "Creating new thought from parent container...",
            # )
            self._current_thought = LLMThought(
                parent_container=self._parent_container,
                expanded=self._expand_new_thoughts,
                collapse_on_complete=self._collapse_completed_thoughts,
                labeler=self._thought_labeler,
            )
            # raise RuntimeError("Current LLMThought is unexpectedly None!")
        return self._current_thought

    def _get_last_completed_thought(self) -> Optional[LLMThought]:
        """Return our most recent completed LLMThought, or None if we don't have one."""
        if len(self._completed_thoughts) > 0:
            return self._completed_thoughts[len(self._completed_thoughts) - 1]
        return None

    @property
    def _num_thought_containers(self) -> int:
        """The number of 'thought containers' we're currently showing: the
        number of completed thought containers, the history container (if it exists),
        and the current thought container (if it exists).
        """
        count = len(self._completed_thoughts)
        if self._history_container is not None:
            count += 1
        if self._current_thought is not None:
            count += 1
        return count

    def _complete_current_thought(self, final_label: Optional[str] = None) -> None:
        """Complete the current thought, optionally assigning it a new label.
        Add it to our _completed_thoughts list.
        """
        thought = self._require_current_thought()
        thought.complete(final_label)
        self._completed_thoughts.append(thought)
        self._current_thought = None

    def _prune_old_thought_containers(self) -> None:
        """If we have too many thoughts onscreen, move older thoughts to the
        'history container.'
        """
        while (
            self._num_thought_containers > self._max_thought_containers
            and len(self._completed_thoughts) > 0
        ):
            # Create our history container if it doesn't exist, and if
            # max_thought_containers is > 1. (if max_thought_containers is 1, we don't
            # have room to show history.)
            if self._history_container is None and self._max_thought_containers > 1:
                self._history_container = MutableExpander(
                    self._history_parent,
                    label=self._thought_labeler.get_history_label(),
                    expanded=False,
                )

            oldest_thought = self._completed_thoughts.pop(0)
            if self._history_container is not None:
                self._history_container.markdown(
                    oldest_thought.container.label)
                self._history_container.append_copy(oldest_thought.container)
            oldest_thought.clear()

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        if self._current_thought is None:
            self._current_thought = LLMThought(
                parent_container=self._parent_container,
                expanded=self._expand_new_thoughts,
                collapse_on_complete=self._collapse_completed_thoughts,
                labeler=self._thought_labeler,
            )

        self._current_thought.on_llm_start(serialized, prompts)

        # We don't prune_old_thought_containers here, because our container won't
        # be visible until it has a child.

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        self._require_current_thought().on_llm_new_token(token, **kwargs)
        self._prune_old_thought_containers()

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        self._require_current_thought().on_llm_end(response, **kwargs)
        self._prune_old_thought_containers()

    def on_llm_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        self._require_current_thought().on_llm_error(error, **kwargs)
        self._prune_old_thought_containers()

    def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
    ) -> None:
        self._require_current_thought().on_tool_start(serialized, input_str, **kwargs)
        self._prune_old_thought_containers()

    def on_tool_end(
        self,
        output: str,
        color: Optional[str] = None,
        observation_prefix: Optional[str] = None,
        llm_prefix: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        self._require_current_thought().on_tool_end(
            output, color, observation_prefix, llm_prefix, **kwargs
        )
        self._complete_current_thought()

    def on_tool_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        self._require_current_thought().on_tool_error(error, **kwargs)
        self._prune_old_thought_containers()

    def on_text(
        self,
        text: str,
        color: Optional[str] = None,
        end: str = "",
        **kwargs: Any,
    ) -> None:
        pass

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> None:
        pass

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        pass

    def on_chain_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        pass

    def on_agent_action(
        self, action: AgentAction, color: Optional[str] = None, **kwargs: Any
    ) -> Any:
        self._require_current_thought().on_agent_action(action, color, **kwargs)
        self._prune_old_thought_containers()

    def on_agent_finish(
        self, finish: AgentFinish, color: Optional[str] = None, **kwargs: Any
    ) -> None:
        if self._current_thought is not None:
            self._current_thought.complete(
                self._thought_labeler.get_final_agent_thought_label()
            )
            self._current_thought = None


##################################
# End Streamlit Callback Handler #
##################################


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
