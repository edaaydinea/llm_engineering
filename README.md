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