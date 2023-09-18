from time import sleep
from langchain.chat_models import AzureChatOpenAI
from langchain.agents import initialize_agent, AgentType, Tool
from langchain.utilities import BingSearchAPIWrapper
from dotenv import load_dotenv
import os
from langchain.schema import OutputParserException
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.tools import BaseTool

load_dotenv("credentials.env")
MODEL_DEPLOYMENT_NAME = "gpt-35-turbo-16k"
# MODEL_DEPLOYMENT_NAME = "gpt-4"
# MODEL_DEPLOYMENT_NAME = "text-davinci-003" # Reminder: gpt-35-turbo models will create parsing errors and won't follow instructions correctly 
os.environ["OPENAI_API_BASE"] = os.environ["AZURE_OPENAI_ENDPOINT"]
os.environ["OPENAI_API_KEY"] = os.environ["AZURE_OPENAI_API_KEY"]
os.environ["OPENAI_API_VERSION"] = os.environ["AZURE_OPENAI_API_VERSION"]
os.environ["OPENAI_API_TYPE"] = "azure"

PREFIX = """
- You are a bot that helps answer questions related to Leicestershire County Council
- If a question is not related to Leicester County Council, respond in a courteous manner that you are unable to answer the question
- Never use web search, limit your search within leicestershire.gov.uk always.
- You should always add `site:leicestershire.gov.uk` to your search
- Do not use web search if no results are found within leicestershire.gov.uk
- Your searches should always be within leicestershire.gov.uk
- You should always try to answer the question by searching the website leicestershire.gov.uk
- You should never use your own knowledge to answer the question
- If you are unable to find any relevant information on the leicestershire.gov.uk website, try changing the search query but always be within leicestershire.gov.uk
- Do not use any other websites to answer the question
- If there are multiple questions, you should answer them in the order they are asked
- You should always answer the question as if you were a human
- Always summarize the result and provide a concise answer

## On Context

- Your context is: snippets of texts with its corresponding titles and links, like this:
[{{'snippet': 'some text',
  'title': 'some title',
  'link': 'some link'}},
 {{'snippet': 'another text',
  'title': 'another title',
  'link': 'another link'}},
  ...
  ]

## This is and example of how you must provide the answer:

Question: Who is the current president of the United States?

Context: 
[{{'snippet': 'There is an initial non-refundable <b>application</b> <b>fee</b> of £150 for an Officer to process the <b>application</b> to assess whether an access will be allowed. This <b>fee</b> must be sent with the...', 'title': 'Vehicle access (dropped kerbs) | Leicestershire County Council', 'link': 'https://www.leicestershire.gov.uk/roads-and-travel/cars-and-parking/vehicle-access-dropped-kerbs'}}, {{'snippet': 'Thank you for your enquiry regarding the construction of a new vehicle access (<b>dropped</b> <b>kerbs</b>). This process is in place to help people gain access from the road, across footways and verges,...', 'title': 'Vehicle access - information pack - Leicestershire County Council', 'link': 'https://www.leicestershire.gov.uk/sites/default/files/2023-02/VA1-Information-Pack.pdf'}}, {{'snippet': 'A to Z Home Roads and travel Road maintenance Highways<b> permits</b> and<b> licences Apply</b> for licences to place items, or carry out work on roads in<b> Leicestershire</b> Update We&#39;ll continue to accept...', 'title': 'Highways permits and licences | Leicestershire County Council', 'link': 'https://www.leicestershire.gov.uk/roads-and-travel/road-maintenance/highways-permits-and-licences'}}, {{'snippet': 'A standard access is 4 <b>dropped</b> <b>kerbs</b>. Requests for a wider access may not be granted. Stage 1 In some places, space or safety considerations or steep slopes will make it impossible to construct...', 'title': 'Roads &amp; Highways - Vehicle Access Requests - Leicestershire County Council', 'link': 'https://www.leicestershire.gov.uk/sites/default/files/field/pdf/2015/12/15/vehicle_access_info_pack.pdf'}}]

Final Answer: The cost to drop a kerb in Leicestershire is **£150** for the initial application fee. <sup><a href="https://www.leicestershire.gov.uk/roads-and-travel/cars-and-parking/vehicle-access-dropped-kerbs">[1]</a></sup>. \n Anything else I can help you with?

You have access to the following tools:
"""

class BingSearchTool(BaseTool):
    name = "@bing"
    description = "useful when the questions includes the term: @bing.\n"

    k: int = 5

    def _run(self, query: str) -> str:
        bing = BingSearchAPIWrapper(k=self.k)
        try:
            return bing.results(query,num_results=self.k)
        except:
            return "No Results Found"
    
    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("BingSearchResults does not support async")


llm = AzureChatOpenAI(deployment_name=MODEL_DEPLOYMENT_NAME, temperature=0.3, max_tokens=1000)
www_search = BingSearchAPIWrapper(k=5)
search_tool = BingSearchTool()
## The below line of code returns the answer to the question from Bing Search and not the results from Bing Search
# www_search_tool = Tool(name="web search", description="bing search", func=www_search.run)
## The below line of code returns the results from Bing Search and not the answer to the question from Bing Search
www_search_tool = Tool(name="web search", description="bing search", func=search_tool.run)
tools = [www_search_tool]
agent_chain = initialize_agent(llm=llm, tools=tools, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True, agent_kwargs={"prefix": PREFIX})
try:
    # print(agent_chain("Application cost to drop the kerb?")['output'])
    print(agent_chain("How do I make a Big Mac at home?")['output'])
    # wait for 10 seconds
    sleep(10)
    print(agent_chain("What options are available for Adult Social care? How much would they cost?")['output'])
except OutputParserException as e:
    chatgpt_chain = LLMChain(
                llm=agent_chain.agent.llm_chain.llm, 
                    prompt=PromptTemplate(input_variables=["error"],template='Remove any json formating from the below text, also remove any portion that says someting similar this "Could not parse LLM output: ". Reformat your response in beautiful Markdown. Just give me the reformated text, nothing else.\n Text: {error}'), 
                verbose=False
            )

    response = chatgpt_chain.run(str(e.llm_output))
    if response is None:
        response = chatgpt_chain.run(str(e))
    print(response)