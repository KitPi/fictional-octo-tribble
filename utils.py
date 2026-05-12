import uuid
from datetime import datetime
from typing import Any, Dict, List

from pydantic import BaseModel


class Item(BaseModel):
    time: datetime
    job_id: uuid.UUID
    data: Any


class ProcessRequest(BaseModel):
    items: List[Item]
