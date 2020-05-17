due-transformers
================

A collection of [Due](https://github.com/dario-chiappetta/Due) Agents that are implemented using [Transformers](https://huggingface.co/transformers/) by Hugging Face.

Setup
-----

Dependencies are managed by Poetry ([installation instructions](https://python-poetry.org/docs/#installation)). This is how you setup the environment:

    poetry install

Demo agent
----------

There currently is only one agent in the library, that uses **Microsoft [DialoGPT](https://huggingface.co/transformers/model_doc/dialogpt.html)**. A console-based example can be run as follows:

    poetry run python run.py

This example code instantiates a DialoGPT agent and serves it on Due's console-based interface:

```python
from due_transformers.dialogpt import DialoGPTAgent
agent = DialoGPTAgent(agent_id='dgpt')

from due.serve import console
console.serve(agent)
```

Project status
--------------

This project is still being worked on, upcoming features will include agent (de)serialization, episode-based training, and more models to be defined.
