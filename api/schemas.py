from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel
from typing import List

class DeckImportRequest(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)
    deck_data: str = Field(..., min_length=1, max_length=5000)

class DeckImportResponse(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)
    main: List[int]
    extra: List[int]
    side: List[int]
