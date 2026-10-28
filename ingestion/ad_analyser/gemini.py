import json
import logging
import mimetypes
import os
import threading
import time
from typing import Any

from google import genai
from google.genai import errors, types

from .asset_staging import stage_file, to_https
from .config import AdAnalyserConfig
from .exceptions import IntegrationError
from .prompts import build_system_instruction
from .storage import StorageManager

logger = logging.getLogger(__name__)

thread_local = threading.local()

class GeminiAIManager:
    def __init__(self, 
                 config: AdAnalyserConfig, 
                 storage_manager: StorageManager, 
                 ai_fields: list[str]):
        self.config = config
        self.storage_manager = storage_manager
        self.ai_fields = ai_fields
        self.system_instruction = config.system_instruction or build_system_instruction(
            self.config.params or {}, 
            self.ai_fields
        )

        self.safety_settings = [
            types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH, 
                                threshold=types.HarmBlockThreshold.BLOCK_NONE),
            types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, 
                                threshold=types.HarmBlockThreshold.BLOCK_NONE),
            types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, 
                                threshold=types.HarmBlockThreshold.BLOCK_NONE),
            types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HARASSMENT, 
                                threshold=types.HarmBlockThreshold.BLOCK_NONE),
        ]

        self.generation_config = types.GenerateContentConfig(
            temperature=0,
            top_p=0.95,
            system_instruction=self.system_instruction,
            safety_settings=self.safety_settings,
            response_mime_type="application/json",
            response_schema=self.config.response_schema,
            labels=self.config.labels
        )
        
        self.http_options = types.HttpOptions(
            retry_options=types.HttpRetryOptions()
        )

    def _get_client(self) -> genai.Client:
        if not hasattr(thread_local, "client"):
            api_key = os.getenv("GEMINI_API_KEY")
            if api_key:
                thread_local.client = genai.Client(
                    api_key=api_key,
                    vertexai=False,
                    http_options=self.http_options
                )
            else:
                thread_local.client = genai.Client(
                    vertexai=True, 
                    project=self.config.gcp_project_id, 
                    location=self.config.gcp_region,
                    http_options=self.http_options
                )
        return thread_local.client

    def _generate_content_from_url(self, url: str) -> types.Part:
        mime_type, _ = mimetypes.guess_type(url)
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            if url.startswith("gs://"):
                try:
                    data = self.storage_manager.read_bytes_from_gcs(url)
                    return types.Part.from_bytes(data=data, mime_type=mime_type or "image/jpeg")
                except Exception:
                    pass
            base_name = os.path.basename(url)
            local_cand = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "assets", "frames", base_name)
            if os.path.exists(local_cand):
                try:
                    with open(local_cand, "rb") as fh:
                        return types.Part.from_bytes(data=fh.read(), mime_type=mime_type or "image/jpeg")
                except Exception:
                    pass
        if mime_type and not url.startswith("gs://"):
            return types.Part.from_uri(file_uri=url, mime_type=mime_type)
        if url.startswith("gs://") and not api_key:
            return types.Part.from_uri(file_uri=url, mime_type=mime_type)
        return self._generate_content_from_gcs(url)

    def _generate_content_from_gcs(self, url: str) -> types.Part:
        gcs_uri, mime_type = stage_file(url, self.storage_manager)
        if gcs_uri:
            api_key = os.getenv("GEMINI_API_KEY")
            if api_key:
                try:
                    data = self.storage_manager.read_bytes_from_gcs(gcs_uri)
                    return types.Part.from_bytes(data=data, mime_type=mime_type or "image/jpeg")
                except Exception:
                    pass
            return types.Part.from_uri(file_uri=gcs_uri, mime_type=mime_type)
        raise IntegrationError(f"Failed to generate content from GCS: {url}")

    def analyze_creative(self, creative_url: str, prompt: str) -> dict[str, Any]:
        creative_url = to_https(creative_url)
        client = self._get_client()
        response = None
        max_empty_retries = 2
        
        for attempt in range(max_empty_retries + 1):
            try:
                content = self._generate_content_from_url(creative_url)
                response = client.models.generate_content(
                    model=self.config.model,
                    contents=[content, prompt],
                    config=self.generation_config
                )   
            except errors.ClientError as e:
                msg = str(e).lower()
                if any(x in msg for x in ["cannot fetch content", 
                                          "robots.txt", 
                                          "url_rejected", 
                                          "invalid argument"]):
                    staged_part = self._generate_content_from_gcs(creative_url)
                    response = client.models.generate_content(
                        model=self.config.model,
                        contents=[staged_part, prompt], 
                        config=self.generation_config
                    )
                else:
                    raise e
            except errors.APIError as e:
                msg = str(e).lower()
                if "429" in msg or "503" in msg or "500" in msg:
                    time.sleep(2)
                    content = self._generate_content_from_url(creative_url)
                    response = client.models.generate_content(
                        model=self.config.model,
                        contents=[content, prompt],
                        config=self.generation_config
                    )
                else:
                    raise e
            except Exception as e:
                msg = str(e).lower()
                if "temporary failure" in msg:
                    time.sleep(2)
                    content = self._generate_content_from_url(creative_url)
                    response = client.models.generate_content(
                        model=self.config.model,
                        contents=[content, prompt],
                        config=self.generation_config
                    )
                else:
                    raise e

            if response and response.text:
                self._record_usage(response)
                return json.loads(response.text)

            if attempt < max_empty_retries:
                time.sleep(2)

        raise IntegrationError("Empty response from model after retries.")

    def _record_usage(self, response) -> None:
        """Log token usage to the cost ledger.

        The SDK reports prompt tokens split by modality; video and image input
        bill at different rates from text, so they are recorded separately
        rather than lumped into one number.
        """
        try:
            from common import costs
        except ImportError:
            return
        um = getattr(response, "usage_metadata", None)
        if um is None:
            return
        stage = (self.config.labels or {}).get("stage", "enrichment")

        by_modality = {}
        for det in (getattr(um, "prompt_tokens_details", None) or []):
            mod = str(getattr(det, "modality", "")).upper()
            by_modality[mod] = by_modality.get(mod, 0) + (getattr(det, "token_count", 0) or 0)

        sku_for = {"VIDEO": "gemini_video_in", "IMAGE": "gemini_image_in",
                   "TEXT": "gemini_text_in", "AUDIO": "gemini_text_in"}
        counted = 0
        for mod, tokens in by_modality.items():
            if tokens:
                costs.record(stage, "gemini", sku_for.get(mod, "gemini_text_in"), tokens)
                counted += tokens

        # Fall back to the total if the SDK gave no modality breakdown, so token
        # counts are never silently dropped.
        total_in = getattr(um, "prompt_token_count", 0) or 0
        if counted < total_in:
            costs.record(stage, "gemini", "gemini_text_in", total_in - counted)

        out = getattr(um, "candidates_token_count", 0) or 0
        thoughts = getattr(um, "thoughts_token_count", 0) or 0
        if out:
            costs.record(stage, "gemini", "gemini_text_out", out)
        if thoughts:
            costs.record(stage, "gemini", "gemini_text_out_thinking", thoughts)
