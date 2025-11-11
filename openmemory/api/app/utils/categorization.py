import logging
from typing import List

from app.utils.prompts import MEMORY_CATEGORIZATION_PROMPT
from dotenv import load_dotenv
from pydantic import BaseModel
from pydantic import SkipValidation

from tenacity import retry, stop_after_attempt, wait_exponential
from app.utils.memory import get_memory_client
from openai.lib._parsing import type_to_response_format_param
from app.utils.try_parse_json_object import try_parse_json_object, GLM_JSON_RESPONSE_PREFIX, GLM_JSON_RESPONSE_SUFFIX, PATTERN
import pydantic
from typing import Annotated, Generic, Optional

load_dotenv() 

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class MemoryCategories(BaseModel):
    categories: List[str]


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=15))
def get_categories_for_memory(memory: str) -> List[str]:
    
        # Try to get memory client safely
    try:
        memory_client = get_memory_client()
        if not memory_client:
            raise Exception("Memory client is not available")
    except Exception as client_error:
        logging.warning(f"Memory client unavailable: {client_error}. Creating memory in database only.")
        # Return a json response with the error
        return {
            "error": str(client_error)
        }

    try:
        
        def _parse_obj(pydantic_object: Annotated[type[pydantic.BaseModel], SkipValidation()],
                       obj: dict) -> pydantic.BaseModel:
            try:
                if issubclass(pydantic_object, pydantic.BaseModel):
                    return pydantic_object.model_validate(obj)
                if issubclass(pydantic_object, pydantic.v1.BaseModel):
                    return pydantic_object.parse_obj(obj)
                msg = f"Unsupported model version for PydanticOutputParser: \
                            {pydantic_object.__class__}"
                raise ValueError(msg)
            except (pydantic.ValidationError, pydantic.v1.ValidationError) as e:
                raise ValueError(f"Failed to parse object: {e}")
            
        format_json = GLM_JSON_RESPONSE_PREFIX.format(format_json=type_to_response_format_param(MemoryCategories))

        messages = [
            {"role": "system", "content": f"{format_json}\r\n{MEMORY_CATEGORIZATION_PROMPT}"},
            {"role": "user", "content": memory + GLM_JSON_RESPONSE_SUFFIX},
        ]
        # Let OpenAI handle the pydantic parsing directly
        response: str = memory_client.llm.generate_response(
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0
        )
 
        try:
            action_match = PATTERN.search(str(response))
            if action_match is  None:
                raise ValueError(f"Invalid extraction from contract:  action_match json GLM_JSON_RESPONSE_SUFFIX")
                
            json_text, json_object = try_parse_json_object(action_match.group(1).strip())
        except Exception as e:
            
            json_text, json_object = try_parse_json_object(str(response))
            
        parsed = _parse_obj(MemoryCategories, json_object)
        return [cat.strip().lower() for cat in parsed.categories]

    except Exception as e:
        logging.error(f"[ERROR] Failed to get categories: {e}")
        try:
            logging.debug(f"[DEBUG] Raw response: {response}")
        except Exception as debug_e:
            logging.debug(f"[DEBUG] Could not extract raw response: {debug_e}")
        raise
