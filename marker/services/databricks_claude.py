import json
import time
from typing import Annotated, List, T

import openai
import PIL
from marker.logger import get_logger
from openai import APITimeoutError, RateLimitError
from PIL import Image
from pydantic import BaseModel

from marker.schema.blocks import Block
from marker.services import BaseService

logger = get_logger()


class DatabricksClaudeService(BaseService):
    databricks_claude_base_url: Annotated[
        str, "The base url to use for Databricks Claude models.  No trailing slash."
    ] = None
    databricks_claude_model: Annotated[str, "The model name to use for Databricks Claude model."] = None
    databricks_claude_api_key: Annotated[
        str, "The API key to use for the Databricks Claude service."
    ] = None
    openai_image_format: Annotated[
        str,
        "The image format to use for the OpenAI-like service. Use 'png' for better compatability",
    ] = "webp"
    rate_limit_fail: Annotated[
        bool, "Whether to fail on rate limit errors instead of waiting."
    ] = False

    def process_images(self, images: List[Image.Image]) -> List[dict]:
        """
        Generate the base-64 encoded message to send to an
        openAI-compatabile multimodal model.

        Args:
            images: Image or list of PIL images to include
            format: Format to use for the image; use "png" for better compatability.

        Returns:
            A list of OpenAI-compatbile multimodal messages containing the base64-encoded images.
        """
        if isinstance(images, Image.Image):
            images = [images]

        img_fmt = self.openai_image_format
        return [
            {
                "type": "image_url",
                "image_url": {
                    "url": "data:image/{};base64,{}".format(
                        img_fmt, self.img_to_base64(img, format=img_fmt)
                    ),
                },
            }
            for img in images
        ]

    def validate_response(self, response_text: str, schema: type[T]) -> T:
        response_text = response_text.strip()
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]

        try:
            # Try to parse as JSON first
            out_schema = schema.model_validate_json(response_text)
            out_json = out_schema.model_dump()
            return out_json
        except Exception:
            try:
                # Re-parse with fixed escapes
                escaped_str = response_text.replace("\\", "\\\\")
                out_schema = schema.model_validate_json(escaped_str)
                return out_schema.model_dump()
            except Exception:
                return

    def get_client(self) -> openai.OpenAI:
        return openai.OpenAI(api_key=self.databricks_claude_api_key, base_url=self.databricks_claude_base_url)

    def __call__(
        self,
        prompt: str,
        image: PIL.Image.Image | List[PIL.Image.Image] | None,
        block: Block | None,
        response_schema: type[BaseModel],
        max_retries: int | None = None,
        timeout: int | None = None,
    ):
        if max_retries is None:
            max_retries = self.max_retries

        if timeout is None:
            timeout = self.timeout

        schema_example = response_schema.model_json_schema()
        system_prompt = f"""
Follow the instructions given by the user prompt.  You must provide your response in JSON format matching this schema:

{json.dumps(schema_example, indent=2)}

Respond only with the JSON schema, nothing else.  Do not include ```json, ```,  or any other formatting.
""".strip()

        client = self.get_client()
        image_data = self.format_image_for_llm(image)

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    *image_data,
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        total_tries = max_retries + 1
        for tries in range(1, total_tries + 1):
            try:
                response = client.chat.completions.create(
                    model=self.databricks_claude_model,
                    messages=messages,
                    timeout=timeout,
                )
                response_text = response.choices[0].message.content
                total_tokens = response.usage.total_tokens
                if block:
                    block.update_metadata(
                        llm_tokens_used=total_tokens, llm_request_count=1
                    )
                return self.validate_response(response_text, response_schema)
            except (APITimeoutError, RateLimitError) as e:
                # Rate limit exceeded
                if tries == total_tries:
                    if self.rate_limit_fail:
                        raise e
                    # Last attempt failed. Give up
                    logger.error(
                        f"Rate limit error: {e}. Max retries reached. Giving up. (Attempt {tries}/{total_tries})",
                    )
                    break
                else:
                    wait_time = tries * self.retry_wait_time
                    logger.warning(
                        f"Rate limit error: {e}. Retrying in {wait_time} seconds... (Attempt {tries}/{total_tries})",
                    )
                    time.sleep(wait_time)
            except Exception as e:
                logger.error(f"Databricks Claude inference failed: {e}")
                break

        return {}
