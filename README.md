# LLM based Solr & Lucene Technical Support Assistant
An LLM based technical Solr assistant which pulls in data from Solr mailing lists, Github PRs and official Solr documentation for context to answer technical user queries.

<h2>How to run</h2>

1) Setup a virtual environment<br/>
	python3 -m venv venv<br/>
	source venv/bin/activate   #or 'venv\Scripts\activate' on Windows

2) pip install -r requirements.txt

3) python3 generate_project_folder_structure.py

4) The tool currently ingests the Github issues from apache/solr and apache/lucene-solr projects for its dataset. User mailing lists and documentation to be added.
You'll need a Github Personal Access Token (PAT) in order to be able to crawl the Github issues (Refer to https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens#creating-a-fine-grained-personal-access-token). 
In the project folder create a .env file with your PAT and OpenAI API Key:<br/>
GITHUB_TOKEN=\<Your PAT\><br/>
OPENAI_API_KEY=\<Your OpenAI API key\>

The initial setup is now complete.



<h2>Scripts to be run from scripts/ folder in the given sequence</h2>

	fetch_github_issues.py ==> Fetches Github issues from apache/solr and apache/lucene-solr, and stores in the data/github_issues folder
	chunk_issues.py ==> Chunks the title+body and PR comments in chunks of 300 tokens with a 20% overlap between chunks for better context retention during vector generation 
	index_chunks.py ==> Generates embeddings using mpnet model (with 384-512 token context window) and indexes them into ChromaDB

<h2>Phases completed</h2>
Phase 1: Ingestion
   
    Goal: Ingest core data sources for retrieval.

Phase 2: Semantic search

    Goal: Generate and Index embeddings in ChromaDB to support semantic search for RAG

Phase 3: Retrieval-Augmented Generation (RAG)

    Goal: Combine retrieval with LLMs to produce grounded answers.

Phase4: Agentic Tooling

    Goal: Evolve RAG into a modular, multi-step “agent” pipeline.

<h3>To-Do (WIP)</h3>

Phase 5: MCP refactor and Server Deployment

	Goal: Turn the agent pipeline into a production-ready service.
