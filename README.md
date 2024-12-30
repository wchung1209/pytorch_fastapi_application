# PyTorch FastAPI Application Demo

This project is a demonstration of a PyTorch FastAPI application tested, built, and deployed to end users. This project is based on learnings from DATASCI255: ML Systems Engineering. 

This repo contains all code for end-to-end functional prediction API, including:

- Dependency management using `poetry`
- Packaging up an existing NLP model from HuggingFace
- `FastAPI` application with endpoints to serve prediction results from user requests
- Endpoint testing with `pydantic`
- Application built and tested locally with `Docker`
- Cache results with `Redis` to protect endpoint from abuse
- Deployment for end users to `Azure` with `Kubernetes`
- Using `k6` to load test and `Grafana` to visualize the dynamics of the system