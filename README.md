# LLM Engineering: Master AI, Large Language Models & Agents

This course provides a comprehensive guide to mastering AI, focusing on Large Language Models (LLMs) and agents. It covers essential concepts, practical coding exercises, and comparisons of leading LLMs to equip participants with the skills needed to build and deploy LLM-based solutions.

[Enroll in the course on Udemy](https://udemy.com/course/llm-engineering-master-ai-and-large-language-models)

## Table of Contents

- [LLM Engineering: Master AI, Large Language Models \& Agents](#llm-engineering-master-ai-large-language-models--agents)
  - [Table of Contents](#table-of-contents)
  - [Week 1 - Build Your First LLM Product: Exploring Top Models \& Transformers](#week-1---build-your-first-llm-product-exploring-top-models--transformers)
    - [Day 1](#day-1)
    - [Day 2](#day-2)
    - [Day 3](#day-3)
    - [Day 4](#day-4)
    - [Day 5](#day-5)
  - [Week 2 - Build a Multi-Modal Chatbot: LLMs, Gradio UI, And Agents](#week-2---build-a-multi-modal-chatbot-llms-gradio-ui-and-agents)
    - [Day 6](#day-6)
    - [Day 7](#day-7)
    - [Day 8](#day-8)
    - [Day 9](#day-9)
    - [Day 10](#day-10)
  - [Week 3: Open-Source Gen AI: Building Automated Solutions with HuggingFace](#week-3-open-source-gen-ai-building-automated-solutions-with-huggingface)
    - [Day 11](#day-11)
    - [Day 12](#day-12)
    - [Day 13](#day-13)
    - [Day 14](#day-14)
    - [Day 15](#day-15)
  - [Week 4 - LLM Showdown: Evaluating Models for Code Generation \& Business Tasks](#week-4---llm-showdown-evaluating-models-for-code-generation--business-tasks)
    - [Day 16](#day-16)
    - [Day 17](#day-17)
    - [Day 18](#day-18)
    - [Day 19](#day-19)

## Week 1 - Build Your First LLM Product: Exploring Top Models & Transformers

### Day 1

**What I did today:**

- Introduced to the course and its objectives.
- Learned about the basics of Large Language Models (LLMs).
- Used Ollama to run LLMs locally.
- Wrote code to call OpenAI's frontier models.
- Distinguished between System and User prompts.
- Learned summarization techniques applicable to many commercial problems.

**Resources**:

- [day1.ipynb](./week1/day1.ipynb)

### Day 2

**What I did today:**

- Reviewed the installation and setup of Ollama.
- Upgraded the Day 1 project to use an open-source model running locally via Ollama.
- Implemented a website summarizer using Llama 3.2.
- Explored alternative approaches using the OpenAI client library to call Ollama.
- Experimented with the DeepSeek reasoning model.

**Resources**:

- [day2 EXERCISE.ipynb](./week1/day2%20EXERCISE.ipynb)
- [day2 notes.ipynb](./week1/notes/day2.ipynb)

### Day 3

**What I did today:**

- Reflected on the capabilities of six leading LLMs, emphasizing their power and convergence in performance.
- Discussed evolving factors that differentiate these models, such as price and specific features.
- Conducted a fun, unscientific leadership challenge between GPT-4, Claude 3 Opus, and Gemini 1.5 Pro.
- Analyzed the pitches made by Alex (GPT-4), Blake (Claude 3 Opus), and Charlie (Gemini 1.5 Pro) for leadership.
- Prepared for the next session, which will delve into the technical aspects of LLMs, including Transformers, tokens, context windows, parameters, and API costs.

**Resources**:

- [day3 notes.ipynb](./week1/notes/day3.ipynb)

### Day 4

**What I did today:**

- Covered essential concepts like tokens, tokenization, context windows, and API costs.
- Clarified the difference between the chat interface cost and the API cost.
- Discussed the challenge of counting letters in a tokenized text.
- Explained why some models were able to answer the "how many A's" question.
- Practiced writing code to call the OpenAI API and local models like Llama.
- Compared and contrasted different frontier LLMs.

**Resources**:

- [day4 notes.ipynb](./week1/notes/day4.ipynb)

### Day 5

**What I did today:**

- Successfully completed the first week, gaining a comprehensive understanding of Transformer models, tokenization techniques, and context window limitations.
- Explored various frontier AI models, assessing their capabilities and constraints in real-world applications.
- Developed practical experience with the OpenAI API, implementing streaming responses and markdown formatting.
- Designed and built a personal AI tutor tool, applying multi-shot prompting to enhance interactions.
- Experimented with system prompts to refine model responses based on tone, character, and instruction adherence.
- Integrated the Llama API to facilitate efficient local model interactions.
- Outlined key objectives for the upcoming week, including multi-model API usage, agent development, and UI implementation with Gradio.

**Resources**:

- [day5.ipynb](./week1/day5.ipynb)
- [day5 notes.ipynb](./week1/notes/day5.ipynb)
- [week1 EXERCISE.ipynb](./week1/week1%20EXERCISE.ipynb)

## Week 2 - Build a Multi-Modal Chatbot: LLMs, Gradio UI, And Agents

### Day 6

**What I did today:**

- Successfully configured API keys for Anthropic (Claude) and Google (Gemini), expanding the toolkit for LLM development.
- Demonstrated the ability to integrate and utilize OpenAI, Anthropic, and Google APIs within a JupyterLab environment, including setting parameters and streaming responses.
- Implemented real-time LLM output by streaming responses from Claude and OpenAI, effectively handling markdown formatting.
- Constructed multi-turn adversarial conversations between GPT-4-mini and Claude-3-haiku, showcasing the manipulation of message lists and system prompts.
- Explored and compared the API structures and functionalities of OpenAI, Claude, and Gemini, highlighting their differences and similarities.
- Applied temperature control to influence the creativity and randomness of LLM outputs, showcasing practical parameter adjustments.
- Designed and executed a joke generation experiment to compare the humor capabilities of different LLMs, providing insights into their creative outputs.
- Reviewed and understood the key components of Transformers, including context windows, tokens, and API costs, reinforcing foundational knowledge.

**Resources**:

- [day1.ipynb](./week2/notebooks/day1.ipynb)
- [day1 notes.ipynb](./week2/notes/day1.ipynb)

### Day 7

**What I did today:*

- Developed proficiency in using Gradio for rapid UI prototyping, specifically for machine learning models and LLMs.
- Implemented basic Gradio interfaces, including text input/output and function integration.
- Learned to share Gradio UIs via local web servers and public URLs.
- Integrated OpenAI's GPT models into Gradio UIs for interactive applications.
- Implemented streaming responses and markdown formatting in Gradio interfaces for enhanced user experience.
- Successfully built a multi-model UI, allowing users to switch between GPT-4, GPT-4-mini, GPT-o1, an GPT-o3-mini models.
- Constructed a company brochure generator application using Gradio, integrating web scraping and LLM API calls.
- Gained experience in creating dynamic and interactive applications with Gradio for LLM-based tasks.
- Prepared for future development of chat UIs and customer support assistants, enhancing prompt context.

**Resources**:

- [day2.ipynb](./week2/notebooks/day2.ipynb)
- [day2 notes.ipynb](./week2/notes/day2.ipynb)

### Day 8

**What I did today:*

- Developed functional chatbot user interfaces using Gradio's `ChatInterface` and OpenAI's API.
- Implemented context management in chatbots by passing the entire conversation history to the LLM with each interaction.
- Utilized system prompts to define chatbot personas, subject matter expertise, and conversation rules.
- Applied one-shot and multi-shot prompting techniques to guide chatbot responses and incorporate dynamic context.
- Constructed OpenAI API message structures, understanding the roles of "system," "user," and "assistant."
- Gained practical experience in converting message structures into tokens for LLM processing, including special tokens.
- Enhanced chatbot functionality by dynamically altering system messages based on user input for improved context.
- Previewed the upcoming exploration of "tools," focusing on empowering LLMs to execute code and perform specific functionalities.

**Resources**:

- [day3.ipynb](./week2/notebooks/day3.ipynb)
- [day3 notes.ipynb](./week2/notes/day3.ipynb)

### Day 9

**What I did today:*

- Implemented and utilized "tools" to enhance LLM capabilities, enabling interaction with external functions.
- Developed a `get_ticket_price` function, demonstrating the ability to integrate custom functions with LLMs for practical applications.
- Constructed a dictionary structure to define function parameters and descriptions, facilitating LLM understanding and usage.
- Handled LLM requests to execute external tools by parsing JSON arguments and returning tool results.
- Built an airline customer service assistant that retrieves ticket prices based on city destinations using the OpenAI API and custom tools.
- Explored the workflow of equipping LLMs with tools, including defining, passing, and handling tool calls.
- Prepared for the next session, which will cover agent development and multi-modality, focusing on complex task handling and image generation.

**Resources**:

- [day4.ipynb](./week2/notebooks/day4.ipynb)
- [day4 notes.ipynb](./week2/notes/day4.ipynb)

### Day 10

**What I did today:**

- Gained an understanding of autonomous software agents and agent frameworks, recognizing their goal-oriented and task-specific nature.
- Learned how agent frameworks facilitate complex problem-solving with minimal human intervention by leveraging various tools.
- Developed a function to generate images using Dall-E 3, exploring its potential for creative image generation from text prompts.
- Integrated text-to-speech functionality using OpenAI's audio API, experimenting with different voice options for audio output.
- Utilized Python libraries such as PIL and Pi Dub for processing generated images and audio.
- Began building a multimodal AI assistant capable of generating both images and audio responses, enhancing user interaction.
- Explored the combination of task breakdown and tool utilization within an agent framework to build a more sophisticated chatbot.
- Integrated a text-to-speech model into the chatbot, enabling it to audibly communicate its responses.
- Implemented a feature where the chatbot triggers an image generation model based on the context of the conversation, such as displaying a city image when discussing ticket prices.
- Started developing a more complex user interface with Gradio to accommodate multimodal interactions, including displaying generated images.
- Reviewed the developed multimodal airline AI assistant and identified key challenges for further enhancement.
- Understood the first challenge involves adding a tool to simulate booking confirmations.
- Recognized the second challenge is to integrate a translation agent using a different LLM (like Claude) to translate responses.
- Identified the third multimodal challenge as incorporating an audio-to-text agent to enable voice input for the AI assistant.
- Prepared for the upcoming week's focus on the open-source LLM ecosystem, including Hugging Face, pipelines, tokenizers, and running inference on open-source models.

**Resources**:

- [day5.ipynb](./week2/notebooks/day5.ipynb)
- [day5 notes.ipynb](./week2/notes/day5.ipynb)

## Week 3: Open-Source Gen AI: Building Automated Solutions with HuggingFace

### Day 11

**What I did today:**

- Gained a foundational understanding of Hugging Face as a key open-source platform for the data science community, encompassing models, datasets, and application deployment.
- Explored the Hugging Face Hub, navigating its extensive collection of over 800,000 models and 200,000 datasets, and understanding the search and filtering functionalities.
- Became familiar with Hugging Face Spaces as a platform for running and sharing AI applications, often built with Gradio or Streamlit, and observed examples of deployed models and leaderboards.
- Successfully set up a personal Hugging Face account and generated an access token with necessary permissions for programmatic interaction with the Hub.
- Acquired practical knowledge of Google Colaboratory (Colab) as a cloud-based Jupyter notebook environment, emphasizing its ease of use, collaboration features, and integration with Google services.
- Learned to manage Colab runtimes, including selecting CPU and various GPU options (T4, A100), and understood the cost implications for different resource utilization.
- Mastered the process of integrating Hugging Face with Google Colab by securely storing and accessing API keys using Colab's "Secrets" feature.
- Executed basic Python code in Colab and verified GPU availability using command-line tools, confirming the environment's readiness for machine learning tasks.
- Witnessed a practical demonstration of running an open-source text-to-image model (Flux) within Google Colab, highlighting the potential of leveraging cloud GPUs for AI applications.
- Understood the upcoming focus on utilizing Hugging Face's different API levels, starting with pipelines, for various AI tasks such as text, image, and audio generation.

**Resources**:

- [day1.ipynb](./week3/notebooks/day1.ipynb)
- [day1 notes.ipynb](./week3/notes/day1.ipynb)

### Day 12

**What I did today:**

- Successfully gained a foundational understanding of the Hugging Face Transformers library, including its dual API design with high-level Pipelines for rapid task execution and lower-level components for custom development.
- Developed hands-on proficiency in using Hugging Face Pipelines for executing a wide range of AI inference tasks with minimal code, significantly simplifying the application of pre-trained models.
- Implemented various Natural Language Processing (NLP) pipelines, covering tasks such as sentiment analysis, named entity recognition (NER), question answering, text summarization, translation, and zero-shot classification.
- Mastered the technique of assigning pipelines to GPU resources (device="cuda") for accelerated model inference and improved computational performance.
- Explored multimodal AI capabilities by generating images from text prompts using the diffusers library in conjunction with models like Stable Diffusion through the Hugging Face ecosystem.
- Achieved practical experience with audio generation, specifically implementing text-to-speech (TTS) pipelines and customizing voice output through the use of speaker embeddings.
- Ensured a functional development environment by successfully installing and managing key Python libraries such as transformers, datasets, and diffusers within a Google Colab setting.
- Learned to effectively customize pipeline operations by selecting specific pre-trained models from the Hugging Face Hub, tailoring solutions to particular task requirements beyond default configurations.
- Completed an intensive learning module focused on Mastering Hugging Face Pipelines, thereby advancing skills in efficient AI inference for a variety of machine learning tasks.

**Resources**:

- [day2.ipynb](./week3/notebooks/day2.ipynb)
- [day2 notes.ipynb](./week3/notes/day2.ipynb)

### Day 13

**What I did today:**

- Gained a comprehensive understanding of tokenizer functionalities within the Hugging Face Transformers library, including the processes of encoding text into numerical tokens and decoding tokens back into human-readable text.
- Mastered the critical concept of model-specificity in tokenizers, recognizing that each transformer model requires its designated tokenizer for accurate inference and to prevent performance degradation.
- Explored essential tokenizer components, such as the vocabulary (vocab), special tokens (e.g., `start of sentence`, `end of sentence`), and chat templates, understanding their role in guiding model behavior and structuring conversational input.
- Investigated the diversity in tokenization strategies across various open-source models, including Llama 3.1, Phi-3, Qwen2 (multilingual), and StarCoder 2 (code generation), noting how these strategies are tailored to model-specific tasks and training data.
- Acquired practical experience in setting up the Hugging Face environment, including API token login and adherence to model-specific Terms of Service for models like Llama 3.1.
- Developed proficiency in using `AutoTokenizer.from_pretrained()` to load appropriate tokenizers, and core methods like `tokenizer.encode()`, `tokenizer.decode()`, and `tokenizer.batch_decode()` for text processing and detailed token inspection.
- Learned to utilize `tokenizer.apply_chat_template()` to correctly format conversational histories for instruct fine-tuned models, ensuring appropriate inclusion of special tokens and model-specific structuring for dialogue.
- Compared the distinct tokenization outputs and chat template structures of Llama 3.1, Phi-3, and Qwen2, reinforcing the necessity of using the correct tokenizer and chat format for each model.
- Examined the specialized nature of tokenizers for domain-specific models like StarCoder 2, observing its optimization for handling programming code syntax and constructs.
- Understood the nuances of tokenization, such as case sensitivity, the handling of spaces, sub-word tokenization, and the typical token-to-character ratio, for more accurate input preparation and analysis.
- Solidified the foundational knowledge required to transition from high-level pipelines to direct interaction with models for advanced text generation tasks.

**Resources**:

- [day3.ipynb](./week3/notebooks/day3.ipynb)
- [day3 notes.ipynb](./week3/notes/day3.ipynb)

### Day 14

**What I did today:**

- Gained a comprehensive understanding of the Hugging Face Model Class, enabling lower-level control over inference processes for open-source transformer models.
- Acquired practical experience in loading, inspecting, and running inference on various open-source models such as Llama 3.1, Phi 3, and Gemma 2, including comparative analysis of their outputs.
- Mastered the application of quantization techniques, specifically using `BitsAndBytesConfig` to load models in 4-bit precision, significantly reducing memory footprint and improving inference speed while observing coherent outputs.
- Developed proficiency in managing the end-to-end text generation pipeline, including tokenizer initialization with `AutoTokenizer`, chat template application, and model loading via `AutoModelForCausalLM`.
- Implemented text generation using `model.generate()` and successfully decoded output tokens back to human-readable text.
- Explored and implemented real-time, token-by-token output streaming using the `TextStreamer` class to enhance user experience in interactive applications.
- Enhanced understanding of model architectures by inspecting underlying PyTorch layers and their dimensions (e.g., vocabulary size, embedding layers, attention mechanisms).
- Practiced essential memory management techniques, including object deletion and GPU cache clearing (`torch.cuda.empty_cache()`), crucial for working with large models in resource-constrained environments.
- Developed a reusable Python function encapsulating model loading, tokenization, quantized inference with streaming, and resource cleanup for efficient experimentation.
- Recognized the importance of model-specific prompting and differences in behavior across various open-source language models.
- Consolidated knowledge of Hugging Face Transformers library, encompassing pipelines, tokenizers, and direct model class interactions, preparing for future projects involving both open-source and frontier model APIs.
  
**Resources**:

- [day4.ipynb](./week3/notebooks/day4.ipynb)
- [day4 notes.ipynb](./week3/notes/day4.ipynb)

### Day 15

**What I did today:**

- Designed an AI system to generate structured meeting minutes from audio recordings, combining frontier and open-source models for a practical business application.
- Leveraged frontier models via API for accurate speech-to-text transcription and open-source models hosted locally via Hugging Face for text summarization and action item extraction.
- Utilized publicly available audio recordings of council meetings from Hugging Face datasets as realistic input data for development.
- Solidified understanding of Hugging Face ecosystem tools, including pipelines, tokenizers, and models for inference with open-source Large Language Models (LLMs).
- Accessed and processed data from Google Drive within a Google Colab environment, using the "Meeting Bank" dataset for the meeting minutes generation project.
- Implemented audio-to-text transcription using OpenAI's `whisper-1` model API.
- Employed the `meta-llama/Meta-Llama-3.1-8B-Instruct` model from Hugging Face for generating structured meeting minutes in Markdown format, guided by specific system and user prompts.
- Applied 4-bit quantization techniques to the Llama 3.1 model, enabling efficient inference on standard Colab GPUs.
- Integrated `TextStreamer` from Hugging Face Transformers to display generated text token by token in real-time, enhancing user experience.
- Successfully formatted the LLM output in Markdown for clear and structured presentation of meeting minutes.
- Defined the end-of-week challenge to build a synthetic test data generator using an open-source model, recognizing its broad applicability.
- Recapped key skills from week 3, including working with frontier models, building complex AI assistants, and integrating diverse model types using Hugging Face tools.
- Previewed upcoming week 4 topics, focusing on LLM selection strategies using leaderboards and arenas, and practical code generation with various models.

**Resources**:

- [day5.ipynb](./week3/notebooks/day5.ipynb)
- [day5 notes.ipynb](./week3/notes/day5.ipynb)

## Week 4 - LLM Showdown: Evaluating Models for Code Generation & Business Tasks

### Day 16

**What I did today:**

- Gained a foundational understanding of task-centric Large Language Model (LLM) selection, emphasizing the evaluation of basic attributes such as open vs. closed-source, technical specifications (parameter count, context window, knowledge cutoff date updated to May 2025), cost implications, operational factors, and licensing terms prior to performance benchmarking.
- Explored the Chinchilla Scaling Law, learning its principle of proportional scaling between model parameters and training data size for optimal LLM performance and efficient resource allocation.
- Became acquainted with a diverse suite of common LLM benchmarks (including ARC, DROP, HellaSwag, MMLU, TruthfulQA, Winogrande, and GSM8K) used to assess varied model capabilities like reasoning, comprehension, and problem-solving.
- Delved into specialized benchmarks, understanding the application of Elo ratings for conversational AI evaluation, HumanEval for Python code generation, and MultiPL-E for assessing multilingual coding proficiency.
- Developed a critical perspective on LLM benchmark limitations, including issues of inconsistent application, narrow scope, challenges in measuring nuanced reasoning, risks of training data leakage, and the problem of models overfitting to specific benchmarks, alongside the emerging concern of potential model awareness during evaluation.
- Learned about "next-level" benchmarks (GPQA, BBH, MATH Level 5, IFEval, MuSR, and MMLU-Pro) designed to more rigorously test advanced LLMs on deeper reasoning, expert knowledge (noting Claude 3.5 Sonnet's 59.4% on GPQA prior to May 2025), and complex instruction following.
- Acquired proficiency in navigating and utilizing the Hugging Face Open LLM Leaderboard, understanding its evolutin to incorporate more challenging benchmarks (GPQA, MMLU-Pro, MuSR, etc., launched around June 2024 contextually) and its features for filtering and ranking open-source models like Qwen2, Phi-3, Llama 3.1, and Mistral.
- Recognized the completion of a significant course portion (40%), establishing a solid groundwork for comparing open-source and proprietary LLMs and understanding the nuances of their evaluation metrics.
- Understood the plan to expand knowledge to a broader landscape of leaderboards, encompassing closed-source models, and to explore real-world commercial LLM applications, aiming to develop a comprehensive strategy for selecting optimal LLMs for specific projects and prototyping.

**Resources**:

- [day1 notes.ipynb](./week4/notes/day1.ipynb)

### Day 17

**What I did today:**

- Broadened understanding of Large Language Model (LLM) evaluation by exploring six essential leaderboards beyond the primary Hugging Face Open LLM Leaderboard, including specialized Hugging Face leaderboards (BigCode, LLM Perf, domain-specific, language-specific), Vellum's Leaderboard, the SEAL Leaderboard, and the LMSYS Chatbot Arena.
- Gained insights into specialized Hugging Face leaderboards, focusing on the BigCode Models Leaderboard for code generation, the LLM Perf Leaderboard for performance metrics (speed, accuracy, memory), and the availability of domain-specific (e.g., Open Medical LLM) and language-specific leaderboards.
- Investigated external leaderboards like Vellum.ai's platform for comparing open-source and closed-source models, including practical metrics like API costs, speed, latency, and context window sizes.
- Explored Scale AI's SEAL Leaderboard, which evaluates models on specialized expert skills such as adversarial robustness, coding, and nuanced instruction following for both open and closed-source LLMs.
- Learned about the LMSYS Chatbot Arena as a platform for evaluating LLMs' conversational abilities through human judgment and an Elo rating system, and understood the value of human preference in assessing interaction quality.
- Reviewed diverse commercial applications of LLMs across industries, including legal (Harvey), talent/recruitment (Nebula.io), legacy code porting (Bloop AI), healthcare (Salesforce Einstein Copilot Health Actions), and education (Khan Academy's Khanmigo), to understand practical model selection context.
- Understood the practical application of LLM evaluation through a new challenge: developing a Python to C++ code conversion tool, which will involve selecting and comparing both frontier and open-source models based on acquired knowledge of leaderboards and benchmarks.
- Solidified the ability to confidently choose appropriate LLMs for specific projects by interpreting results from various leaderboards, arenas, and understanding the importance of factors like performance on coding benchmarks (HumanEval, MultiPL-E), instruction following (IFEval), and context window size.

**Resources:**

- [day2 notes.ipynb](./week4/notes/day2.ipynb)

### Day 18

**What I did today:**

- Initiated a project to build a Python to C++ code conversion tool, focusing first on leveraging frontier LLMs (GPT-4o and Claude 3.5 Sonnet) to enhance runtime performance, using a Pi calculation script via the Leibniz formula as the initial test case.
- Set up the development environment in JupyterLab for the code conversion challenge, selected GPT-4o and Claude 3.5 Sonnet based on coding leaderboards (Vellum.ai, Scale AI SEAL), and engineered detailed system and user prompts, including model-specific hints for GPT-4o regarding C++ type handling and header inclusions.
- Developed Python utility functions to interact with OpenAI and Anthropic APIs for code generation, including features for streaming responses and saving the generated C++ code.
- Successfully translated a Python Pi calculation script to C++ using both GPT-4 and Claude, achieving a significant speed-up (approx. 40x, from 8.57s to 0.21s) with the C++ versions compiled and executed.
- Encountered and analyzed a code generation failure where GPT-4's C++ conversion of a "maximum subarray sum" Python script (with a custom LCG for reproducibility) produced an incorrect result, likely due to an integer overflow, highlighting the need for careful validation.
- Observed Claude's superior performance in the "maximum subarray sum" challenge, where it not only generated correct C++ code but also, on a second attempt, re-implemented the solution using a highly efficient single-loop algorithm (Kadane's), achieving a ~13,000x speedup (27s to 2ms) over the brute-force Python version.
- Developed a Gradio user interface for the Python to C++ code converter, allowing users to input Python code, select between GPT and Claude models, and view the streamed C++ translation in real-time, including output cleaning.
- Enhanced the Gradio prototype UI to include functionality for executing both the input Python code and the AI-generated C++ code directly within the interface, displaying their outputs and execution times for immediate comparison, and utilized more extensive C++ compiler optimization flags.
- Conducted final comparative tests on the "Python Hard" (maximum subarray sum) challenge using the enhanced UI, confirming GPT-4's difficulties and observing Claude's ability to provide both a correct direct translation (0.6s) and a remarkably optimized algorithmic solution (0.4ms) on subsequent attempts.

**Resources:**

- [day3 notes.ipynb](./week4/notes/day3.ipynb)

### Day 19

**What I did today:**

- Explored the use of open-source LLMs, particularly CodeQwen, for code generation tasks and their deployment via Hugging Face Inference Endpoints.
- Mastered interaction with deployed models using the `InferenceClient` from `huggingface_hub`, managing tokenization and chat templating.
- Developed a Gradio UI to compare the performance of CodeQwen, GPT-4, and Claude in Python to C++ code conversion.
- Assessed the capabilities and limitations of a 7B parameter open-source CodeQwen model against larger frontier models in handling complex code generation tasks.
- Acquired skills in selecting appropriate LLMs for code generation, understanding the trade-offs between open-source and frontier models based on task complexity and resource constraints.
- Deepened prompt engineering skills specific to code generation by exploring techniques to influence LLM behavior.
- Leveraged the Big Code Models Leaderboard for informed model selection, prioritizing performance metrics for optimal results.
- Recapped key concepts, including efficient code generation via various models and the significance of parameter counts in model performance.

**Resources:**

- [day4 notes.ipynb](./week4/notes/day4.ipynb)
