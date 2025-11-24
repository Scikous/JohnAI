# JohnAI
The aim of this project is both to give a good starting point for anyone to create their own custom AI Johns (technically won't have to be a John), and be something fun for me to build and potentially have my work live on and be incorporated into numerous other projects for a long time to come.

## Table of Contents

* [JohnAI](#johnai)
    * [Features](#features)
* [Setup](#setup)
    * [Quick Start](#quick-start)
    * [Livechat Setup (Optional)](#livechat-setup-optional)
* [Testing](#testing)
* [Acknowledgements](#acknowledgements)

# Features
:information_source: WIP
* Speak to your LLM using STT (see acknowledgements)

# Setup
This is developed and tested on Python 3.12.3. Ubuntu 24.04 LTS on Wayland with an NVIDIA GPU.

:exclamation: This mainly works on Linux (Ubuntu 24.04 LTS, other distros may work). Direct Windows support has been dropped -- only works through WSL2 OR available in the Windows-legacy branch (heavily out-of-date but technically functional-ish).


## Quick Start
:exclamation: Heavy WIP -- You'll need to manually install many a packages currently.

```
uv venv --python 3.12 --seed --system-site-packages
source .venv/bin/activate
```

```
uv pip install poetry
poetry config virtualenvs.in-project true
poetry install
```

:warning: Assumes you're using CUDA 12.8
```
uv pip install -U vllm \
    --torch-backend=cu128 \
    --extra-index-url https://wheels.vllm.ai/nightly
```

After activating the venv, run the following in the root directory:
```
python -m src.process_orchestrator
```

## Livechat Setup (Optional)
:information_source: This is purely optional, only necessary if you want the AI to interact with livechat on YouTube/Twitch/Kick.

:information_source: More information at the related [**Wiki Page**](https://github.com/Scikous/JohnAI/wiki/Livechat-API-Information)

Create a **.env** file inside of the root directory and add as much of the following as desired:
```
#For all
CONVERSATION_LOG_FILE=path/to/conversation_log.csv
HIGH_CHAT_VOLUME=False #Many viewers typing into chat(s)

#For YouTube
YT_FETCH=False
YT_OAUTH2_JSON=client_secret.json
YT_CHANNEL_ID=YourYTChannelID
LAST_NEXT_PAGE_TOKEN=HandledAutomatically

#For Twitch
TW_FETCH=False
TW_CHANNEL=YourChannelName
TW_OWNER_ID=HandledAutomatically
TW_BOT_NAME=YourBotAccountName
TW_BOT_ID=HandledAutomatically
TW_CLIENT_ID=YourClientID
TW_CLIENT_SECRET=YourClientSecret

#For Kick
KI_FETCH=False
KI_CHANNEL=YourChannelName
```

# Testing
:warning: HEAVILY OUT OF DATE DO NOT USE FOR GOD'S SAKE Running just **pytest** will result in running all test files in the entire project which will inevitably fail. We are only concerned with the tests within the tests directory at the root. 

To run all tests use:
```
pytest -s tests
```

Only unit tests:
```
pytest -s tests -m "not integration"
```

Only integration tests
```
pytest -s tests -m integration
```

# Acknowledgements
This project makes use of the following (Out-Of-Date? Fix Later):

* [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS/tree/main)
* [Unsloth](https://github.com/unslothai/unsloth)
* [Dao-AILab](https://github.com/Dao-AILab/flash-attention)
* [turboderp](https://github.com/turboderp/exllamav2)
* [RealtimeTTS](https://github.com/KoljaB/RealtimeTTS)
* [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
