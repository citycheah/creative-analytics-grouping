from pydantic import BaseModel
from typing import List

class CreativeConceptSchema(BaseModel):
    core_concept_summary: str
    messaging_angle: str
    visual_style_tags: List[str]

# Alias for backward compatibility
AdAnalysisResults = CreativeConceptSchema
