# Python notebooks 📒

이 폴더에는 다양한 실행 가능한 Python 노트북이 포함되어 있으므로 모든 것을 직접 테스트할 수 있습니다. 이러한 예를 자체 데이터 테스트를 위한 청사진으로 사용하세요.

[Jupyter](https://jupyter.org/install)를 사용하여 로컬에서 노트북을 실행하거나 각 노트북에 제공된 링크를 사용하여 Google [Colab](https://colab.research.google.com)에서 실행하는 것이 가장 이상적이지만, 여기서는 온프라미스(onpromise) 환경을 전제로 구성되었습니다.

노트북은 다음 폴더로 구성됩니다.

- [`search`](./search/): 임베딩 인덱싱, 어휘, 의미 및 _하이브리드_ 검색 실행 등과 같은 Elasticsearch의 기본 사항을 보여주는 노트북입니다.

- [`generative-ai`](./generative-ai/): LLM 기반 애플리케이션을 위한 검색 엔진 및 벡터 저장소로서 Elasticsearch의 다양한 사용 사례를 보여주는 노트북입니다.

- [`integrations`](./integrations/): 인기 있는 서비스 및 프로젝트를 Elasticsearch와 통합하는 방법을 보여주는 노트북 입니다
  - [OpenAI](./integrations/openai)
  - [Hugging Face](./integrations/hugging-face)
  - [LlamaIndex](./integrations/llama-index)

- [`langchain`](./langchain/): Elastic을 언어 모델로 구동되는 애플리케이션 개발용 프레임워크인 [LangChain](https://langchain-langchain.vercel.app/docs/get_started/introduction.html)과 통합하는 방법을 보여주는 노트북입니다.
