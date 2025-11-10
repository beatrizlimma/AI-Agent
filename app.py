from smolagents import CodeAgent,DuckDuckGoSearchTool, HfApiModel,load_tool,tool
import datetime
import requests
import pytz
import yaml
from tools.final_answer import FinalAnswerTool

from Gradio_UI import GradioUI

@tool
def convert_ml_to_cups(ml: float) -> str:
    """Converts milliliters (ml) to cups
    Args:
        ml: The value in milliliters to be converted to cups
    """
    cups = ml / 240  # 1 cup = 240 ml
    return f"{ml} ml is equivalent to {cups:.2f} cups."

from smolagents import tool
import requests

@tool
def get_country_info(country_name: str) -> str:
    """Returns the flag and official currency of a country
    
    Args:
        country_name: Country name (in Portuguese or English, e.g., 'Brazil', 'Canada', 'Japan').
    """
    try:
        # Makes the request to the Countries REST API
        url = f"https://restcountries.com/v3.1/name/{country_name}?fullText=true"
        response = requests.get(url)
        
        if response.status_code != 200:
            #search by part of the name
            url = f"https://restcountries.com/v3.1/name/{country_name}"
            response = requests.get(url)
        
        if response.status_code != 200:
            return f" Couldn't find any information about the country. '{country_name}'."
        
        data = response.json()[0]
        

        flag = data.get("flag", "🏳️")
        currencies = data.get("currencies", {})
        
        if currencies:
            
            currency_code, currency_data = list(currencies.items())[0]
            currency_name = currency_data.get("name", "Moeda desconhecida")
            currency_info = f"{currency_name} ({currency_code})"
        else:
            currency_info = "Currency information not available"
        
        country_official = data.get("name", {}).get("common", country_name.title())

        return f" Country: {country_official}\n{flag} Flag: {flag}\n Currency: {currency_info}"

    except Exception as e:
        return f" An error occurred while retrieving country information '{country_name}': {str(e)}"


@tool
def translate_text(text: str, target_lang: str)-> str: 
    #Keep this format for the description / args / args description but feel free to modify the tool
    """Simplified translation without API call
    Args:
        text: The text to be translated.
        target_lang: The target language code (e.g., 'pt' for Portuguese, 'ja' to japanese, 'en' to english, 'it' to italian, 'es' to spanish, 'fr' to france, 'de' to german).
    """
    return f"Translation of '{text}' to {target_lang}"

@tool
def get_current_time_in_timezone(timezone: str) -> str:
    """A tool that fetches the current local time in a specified timezone.
    Args:
        timezone: A string representing a valid timezone (e.g., 'America/New_York').
    """
    try:
        # Create timezone object
        tz = pytz.timezone(timezone)
        # Get current time in that timezone
        local_time = datetime.datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
        return f"The current local time in {timezone} is: {local_time}"
    except Exception as e:
        return f"Error fetching time for timezone '{timezone}': {str(e)}"


final_answer = FinalAnswerTool()

# If the agent does not answer, the model is overloaded, please use another model or the following Hugging Face Endpoint that also contains qwen2.5 coder:
# model_id='https://pflgm2locj2t89co.us-east-1.aws.endpoints.huggingface.cloud' 

model = HfApiModel(
max_tokens=2096,
temperature=0.5,
model_id='Qwen/Qwen2.5-Coder-32B-Instruct',# it is possible that this model may be overloaded
custom_role_conversions=None,
)


# Import tool from Hub
image_generation_tool = load_tool("agents-course/text-to-image", trust_remote_code=True)

with open("prompts.yaml", 'r') as stream:
    prompt_templates = yaml.safe_load(stream)
    
agent = CodeAgent(
    model=model,
    tools=[final_answer], ## add your tools here (don't remove final answer)
    max_steps=6,
    verbosity_level=1,
    grammar=None,
    planning_interval=None,
    name=None,
    description=None,
    prompt_templates=prompt_templates
)


GradioUI(agent).launch()