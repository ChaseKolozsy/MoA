import os
import json
import time
import requests
import openai
import copy
from enum import Enum
import google.generativeai as genai

from loguru import logger
from dotenv import load_dotenv

load_dotenv()

class APIProvider(Enum):
    TOGETHER = "together"
    GROQ = "groq"
    DEEPSEEK = "deepseek"
    OPENAI = "openai"  # Add OpenAI to the APIProvider enum
    ANTHROPIC = "anthropic"  # Add Anthropic to the APIProvider enum
    GEMINI = "gemini"  # Add Gemini to the APIProvider enum

DEBUG = int(os.environ.get("DEBUG", "0"))


def generate_together(
    model,
    messages,
    max_tokens=2048,
    temperature=0.7,
    streaming=False,
    api_provider=APIProvider.TOGETHER,
):

    output = None

    for sleep_time in [1, 2, 4, 8, 16, 32]:

        try:

            if api_provider == APIProvider.TOGETHER:
                endpoint = "https://api.together.xyz/v1/chat/completions"
                api_key = os.environ.get("TOGETHER_API_KEY")
            elif api_provider == APIProvider.GROQ:
                endpoint = "https://api.groq.com/openai/v1/chat/completions"
                api_key = os.environ.get("GROQ_API_KEY")
            elif api_provider == APIProvider.OPENAI:
                endpoint = "https://api.openai.com/v1/chat/completions"
                api_key = os.environ.get("OPENAI_API_KEY")
            elif api_provider == APIProvider.DEEPSEEK:
                endpoint = "https://api.deepseek.com/v1/chat/completions"
                api_key = os.environ.get("DEEPSEEK_API_KEY")
            else:
                raise ValueError(f"Unsupported API provider: {api_provider}")

            if DEBUG:
                logger.debug(
                    f"Sending messages ({len(messages)}) (last message: `{messages[-1]['content'][:20]}...`) to `{model}`."
                )

            res = requests.post(
                endpoint,
                json={
                    "model": model,
                    "max_tokens": max_tokens,
                    "temperature": (temperature if temperature > 1e-4 else 0),
                    "messages": messages,
                },
                headers={
                    "Authorization": f"Bearer {api_key}",
                },
            )
            if "error" in res.json():
                logger.error(res.json())
                if res.json()["error"]["type"] == "invalid_request_error":
                    logger.info("Input + output is longer than max_position_id.")
                    return None

            output = res.json()["choices"][0]["message"]["content"]

            break

        except Exception as e:
            logger.error(e)
            if DEBUG:
                logger.debug(f"Msgs: `{messages}`")

            logger.info(f"Retry in {sleep_time}s..")
            time.sleep(sleep_time)

    if output is None:

        return output

    output = output.strip()

    if DEBUG:
        logger.debug(f"Output: `{output[:20]}...`.")

    return output


def generate_together_stream(
    model,
    messages,
    max_tokens=2048,
    temperature=0.7,
    api_provider=APIProvider.TOGETHER,
):
    if api_provider == APIProvider.TOGETHER:
        endpoint = "https://api.together.xyz/v1"
        api_key = os.environ.get("TOGETHER_API_KEY")
    elif api_provider == APIProvider.GROQ:
        endpoint = "https://api.groq.com/openai/v1"
        api_key = os.environ.get("GROQ_API_KEY")
    elif api_provider == APIProvider.DEEPSEEK:
        endpoint = "https://api.deepseek.com/v1"
        api_key = os.environ.get("DEEPSEEK_API_KEY")
    elif api_provider == APIProvider.OPENAI:
        endpoint = "https://api.openai.com/v1"
        api_key = os.environ.get("OPENAI_API_KEY")
    elif api_provider == APIProvider.ANTHROPIC:
        endpoint = "https://api.anthropic.com/v1"
        api_key = os.environ.get("ANTHROPIC_API_KEY")
    else:
        raise ValueError(f"Unsupported API provider: {api_provider}")

    client = openai.OpenAI(
        api_key=api_key, base_url=endpoint
    )
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature if temperature > 1e-4 else 0,
        max_tokens=max_tokens,
        stream=True,  # this time, we set stream=True
    )

    return response


def generate_openai(
    model,
    messages,
    max_tokens=2048,
    temperature=0.7,
):

    client = openai.OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
    )

    for sleep_time in [1, 2, 4, 8, 16, 32]:
        try:

            if DEBUG:
                logger.debug(
                    f"Sending messages ({len(messages)}) (last message: `{messages[-1]['content'][:20]}`) to `{model}`."
                )

            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            output = completion.choices[0].message.content
            break

        except Exception as e:
            logger.error(e)
            logger.info(f"Retry in {sleep_time}s..")
            time.sleep(sleep_time)

    output = output.strip()

    return output


def inject_references_to_messages(
    messages,
    references,
):

    messages = copy.deepcopy(messages)

    system = f"""You have been provided with a set of responses from various open-source models to the latest user query. Your task is to synthesize these responses into a single, high-quality response. It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect. Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the instruction. Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability.

Responses from models:"""

    for i, reference in enumerate(references):

        system += f"\n{i+1}. {reference}"

    if messages[0]["role"] == "system":

        messages[0]["content"] += "\n\n" + system

    else:

        messages = [{"role": "system", "content": system}] + messages

    return messages


def generate_with_references(
    model,
    messages,
    references=[],
    max_tokens=8192,
    temperature=0.7,
    generate_fn=generate_together,
    api_provider=APIProvider.TOGETHER,
):
    if len(references) > 0:
        messages = inject_references_to_messages(messages, references)

    if api_provider == APIProvider.GEMINI:
        return generate_gemini(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    elif api_provider == APIProvider.OPENAI:
        return generate_openai(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    else:
        return generate_fn(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            api_provider=api_provider,
        )


def generate_gemini(
    model,
    messages,
    max_tokens=2048,
    temperature=0.7,
    streaming=False,
    api_provider=APIProvider.GEMINI,
):
    genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
    gemini_model = genai.GenerativeModel(model)

    prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])

    output = None

    for sleep_time in [1, 2, 4, 8, 16, 32]:
        try:
            if DEBUG:
                logger.debug(
                    f"Sending messages ({len(messages)}) (last message: `{messages[-1]['content'][:20]}`) to `{model}`."
                )

            if streaming:
                response = gemini_model.generate_content(
                    prompt,
                    generation_config={
                        "temperature": temperature,
                        "max_output_tokens": max_tokens,
                    },
                    stream=True,
                )
                return response  # Return the streaming response directly
            else:
                response = gemini_model.generate_content(
                    prompt,
                    generation_config={
                        "temperature": temperature,
                        "max_output_tokens": max_tokens,
                    }
                )
                output = response.text
                break

        except Exception as e:
            logger.error(e)
            logger.info(f"Retry in {sleep_time}s..")
            time.sleep(sleep_time)

    if output is None:
        return output

    output = output.strip()

    return output
