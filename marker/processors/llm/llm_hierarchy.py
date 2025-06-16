from __future__ import annotations
import time
from typing import List
from pydantic import BaseModel

from marker.schema import BlockTypes
from marker.schema.blocks import Block
from marker.schema.document import Document
from marker.processors.llm import BaseLLMProcessor

class DocumentItem(BaseModel):
    block_id: str
    parent_block_id: str | None

class LLMHierarchySchema(BaseModel):
    document: List[DocumentItem]

class LLMHierarchyProcessor(BaseLLMProcessor):
    # max_block_text_len: int = 50
    llm_hierarchy_prompt = """
Your task is to analyze the semantic structure of a legal document and output a **flat JSON list**, where each block is represented as an object that includes:

- `block_id`: The original ID of the block.
- `parent_block_id`: A str of ID indicating the **immediate** semantic parent of this block. If the block is a top-level section, this will be None

## Input Description

You will be provided with the identified blocks of a document, where each block is identified by a unique `block_id`. These follow the format `/page/<page_number>/<block_type>/<block_index>`.
You will also be provided with a **snippet** of the text inside of each block, for context.
Example:
```
/page/0/SectionHeader/0:
Provisions of the Agreement

/page/0/Text/1:
Provision 1: 1st provision

/page/0/Text/2:
Provision a: Sub provision of 1st provision

...
```

## Output Requirements
Your output **must be a single JSON array**. Each element in the array represents a document block with the following structure:

```json
{
    "document": [
        {
            "block_id": "/page/0/SectionHeader/0",
            "parent_block_id": null
        },
        {
            "block_id": "/page/0/Text/1",
            "parent_block_id": "/page/0/SectionHeader/0"
        },
        {
            "block_id": "/page/0/Text/2",
            "parent_block_id": "/page/0/Text/1"
        },
        ...
    ]
}
```

**Key principles for semantic nesting:**
* If a block is nested semantically under others (like "1.1 Definitions" under "1. Software License"), the parent id should be that of the immediate parent, and that parent should in turn be labelled with the correct parent id, and so on.
* **Section Headers:** Blocks identified as `SectionHeader` typically introduce new semantic sections and will serve as parent id for multiple following blocks
* **Content Following Headers:** Any `Text` or `ListGroup` blocks that logically fall under a preceding `SectionHeader` should have that header as the parent id
* **Numbered Sections/Subsections:** Pay close attention to numbering schemes (e.g., "1. Software License.", "1.1 Definitions.", "1.2 License Grant."). These indicate hierarchical relationships. Blocks pertaining to a subsection (e.g., "1.1 Definitions.") should be nested under their parent section (e.g., "1. Software License.").
* **Nested Lists:** If there are mulitple levels of bullets - Such as Bullet 1, Sub bullets a and b, and Bullet 2, then bullets a and b should have bullet 1 as the parent
* **Tables and Pictures:** If a `Table` or `Picture` block is directly associated with a specific semantic section (e.g., a table listing licensed software under a heading), it should be nested under that section.

## Input

"""
    def formatted_block(self, block: Block, document: Document):
      block_raw_text = block.raw_text(document)
      return f"{block.id}:\n{block_raw_text[:20]}" + "\n\n"

    def __call__(self, document: Document, **kwargs):
      if not self.use_llm or self.llm_service is None:
          return

      text = "\n"
      for page in document.pages:
          for block in page.structure_blocks(document):
              if block.ignore_for_output:
                  continue
              text += self.formatted_block(block, document)

      response = self.llm_service(self.llm_hierarchy_prompt + text, None, None, LLMHierarchySchema)
      document.llm_hierarchy = response['document']