# 🥑 Guac-AI-Mole

[Demo](https://guac-ai-mole.streamlit.app/) will provide samples questions and answers generated by Guac-AI-Mole!

> 🧪 This is a hackathon project. Do not use in production.

## Development Setup

### Pre-requisites
- Install and run [GUAC](https://docs.guac.sh/setup/)
- Install [Steamlit](https://docs.streamlit.io/library/get-started/installation)

### Run the app
- Install python dependencies with `pip install -r requirements.txt`
- Run `streamlit run app.py` to start the Streamlit app (add `--logger.level=debug` for debug logs)
- Navigate to app URL (default: http://localhost:8501)
- Set up Azure OpenAI API Key, endpoint and deployment name in the sidebar on the left
- Set up GUAC GraphQL endpoint in the sidebar on the left (default: http://localhost:8080/graphql)