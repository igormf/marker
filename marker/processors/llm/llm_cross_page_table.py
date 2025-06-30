from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Annotated, List, Tuple, Literal, Dict, Any, Optional
from dataclasses import dataclass

from pydantic import BaseModel
from tqdm import tqdm
from PIL import Image

from marker.output import json_to_html
from marker.processors.llm import BaseLLMComplexBlockProcessor
from marker.schema import BlockTypes
from marker.schema.blocks import Block, TableCell
from marker.schema.document import Document


class CellLocationSchema(BaseModel):
    table_index: int
    row: int
    col: int

class CorrectionItemSchema(BaseModel):
    type: Literal["merge_cells", "move_cell", "remove_cell", "move_text_fragment"]
    source: CellLocationSchema
    target: Optional[CellLocationSchema] = None
    action: Literal["append", "prepend", "replace"] = "append"
    text_fragment: Optional[str] = None  # For move_text_fragment type

class CorrectionSchema(BaseModel):
    analysis: str
    corrections: List[CorrectionItemSchema]


@dataclass
class CellLocation:
    """Represents a cell location in a table"""
    table_index: int  # 0 for first table, 1 for second table
    row: int
    col: int


@dataclass 
class CellCorrection:
    """Represents a single cell correction operation"""
    type: Literal["merge_cells", "move_cell", "remove_cell", "move_text_fragment"]
    source: CellLocation
    target: Optional[CellLocation] = None
    action: Literal["append", "prepend", "replace"] = "append"
    text_fragment: Optional[str] = None


class LLMCrossPageTableProcessor(BaseLLMComplexBlockProcessor):
    """Generic processor for fixing cell boundary errors in cross-page tables"""
    
    block_types: Annotated[
        Tuple[BlockTypes],
        "The block types to process.",
    ] = (BlockTypes.Table, BlockTypes.TableOfContents)
    
    table_height_threshold: Annotated[
        float,
        "The minimum height ratio relative to the page for the first table in a pair to be considered for correction.",
    ] = 0.6
    
    max_column_difference: Annotated[
        int,
        "Maximum allowed column count difference between table segments to attempt correction",
    ] = 3
    
    disable_tqdm: Annotated[
        bool,
        "Whether to disable the tqdm progress bar.",
    ] = False
    
    correction_prompt: Annotated[
        str,
        "The prompt to use for table correction analysis.",
    ] = """You are a table correction expert specializing in fixing cell boundary errors that occur when tables span across PDF page breaks.

You will receive two images showing table segments from consecutive pages, along with their current HTML representations. Your task is to identify and specify corrections for cell boundary errors.

**FOCUS: PAGE BREAK ISSUES ONLY**
This processor specifically handles content that was incorrectly split due to PDF page breaks. Only fix issues where:
1. **Text split across pages**: Content that should be together but got separated by page boundary
2. **Orphaned fragments**: Single words or phrases in Table 2 that clearly belong to incomplete entries in Table 1
3. **Page break artifacts**: Content misplacement that only occurred because of page boundaries

**DO NOT FIX:**
- Complete, properly formatted entries in either table
- General table formatting issues unrelated to page breaks
- Content that belongs in separate tables/sections

**Instructions:**
1. **Focus on page boundary**: Examine the last few rows of Table 1 and first few rows of Table 2
2. **Look for incomplete entries**: Find entries in Table 1 that appear cut off or incomplete
3. **Find orphaned text**: Look for text in Table 2 that seems to complete those incomplete Table 1 entries
4. **Use append action**: When moving text from Table 2 to Table 1, always use "append" to add to the end
5. **One correction maximum**: Make at most one correction for the clearest page break split

**CRITICAL CONSTRAINT:**
Only move content that was clearly split by the page break. If both tables look complete and separate, make NO corrections.

**Output Format:**
Return a JSON object with corrections. Each correction specifies:
- `type`: "merge_cells" (combine content), "move_cell" (relocate content), "remove_cell" (delete empty cell), or "move_text_fragment" (move specific text)
- `source`: Location of content to be moved/merged (table_index, row, col)
- `target`: Where content should go (for merge_cells, move_cell, and move_text_fragment)
- `action`: How to combine content ("append", "prepend", "replace")
- `text_fragment`: For "move_text_fragment" type, specify the exact text to move

**Page Break Fix Example:**
```json
{
  "analysis": "Looking at the page boundary, Table 1's last row contains 'ABC Software Suite' which appears incomplete, and Table 2's first row contains 'Tools' which looks like the continuation that was split by the page break.",
  "corrections": [
    {
      "type": "move_text_fragment",
      "source": {"table_index": 1, "row": 0, "col": 0},
      "target": {"table_index": 0, "row": 4, "col": 0},
      "action": "append",
      "text_fragment": "Tools"
    }
  ]
}
```

**No Page Break Issues Example:**
```json
{
  "analysis": "Both tables appear to be complete and separate entities. Table 1 ends properly and Table 2 begins with complete entries. No obvious page break artifacts detected.",
  "corrections": []
}
```

**CRITICAL**: When looking at the tables, identify which entries appear incomplete and need the orphaned text. Look for product names, descriptions, or data that seem truncated and need completion from other cells.

**ROW IDENTIFICATION CRITICAL**: 
- Each `<tr>` element has an id attribute showing the row number (e.g., `<tr id="row-0">`, `<tr id="row-1">`, etc.)
- Use these IDs to identify the exact row number: row-0 = 0, row-1 = 1, row-10 = 10, etc.
- Find incomplete product names by looking for entries that seem cut off
- Target the exact row number from the id attribute, NOT nearby complete entries

**Table 1 HTML:**
```html
{{table1_html}}
```

**Table 2 HTML:**
```html
{{table2_html}}
```

Analyze the images and HTML, then provide corrections for any cell boundary errors you identify. Remember to be surgical - only move the specific orphaned text, not entire cell contents."""

    def rewrite_blocks(self, document: Document):
        """Main entry point for processing cross-page table corrections"""
        pbar = tqdm(desc=f"{self.__class__.__name__} running", disable=self.disable_tqdm)
        
        # Find consecutive table pairs that might need correction
        table_pairs = self.find_cross_page_table_pairs(document)
        
        # with ThreadPoolExecutor(max_workers=self.max_concurrency) as executor:
        #     for future in as_completed([
        #         executor.submit(self.process_table_pair, document, table1, table2)
        #         for table1, table2 in table_pairs
        #     ]):
        #         future.result()  # Raise exceptions if any occurred
        #         pbar.update(1)
        for table1, table2 in table_pairs:
            self.process_table_pair(document, table1, table2)
        
        pbar.close()

    def find_cross_page_table_pairs(self, document: Document) -> List[Tuple[Block, Block]]:
        """Find pairs of tables on consecutive pages that might need correction"""
        pairs = []
        
        prev_table = None
        for page in document.pages:
            page_tables = page.contained_blocks(document, self.block_types)
            
            for table in page_tables:
                if prev_table and prev_table.page_id == table.page_id - 1:
                    # Check if this looks like a cross-page table scenario
                    if self.should_check_table_pair(document, prev_table, table):
                        pairs.append((prev_table, table))
                
                prev_table = table
        
        return pairs
    
    def should_check_table_pair(self, document: Document, table1: Block, table2: Block) -> bool:
        """Determine if a table pair should be checked for corrections"""
        cells1 = table1.contained_blocks(document, (BlockTypes.TableCell,))
        cells2 = table2.contained_blocks(document, (BlockTypes.TableCell,))
        
        if not cells1 or not cells2:
            return False
        
        # Get column counts
        col_count1 = self.get_column_count(cells1)
        col_count2 = self.get_column_count(cells2)
        
        # Check if column counts are close enough to suggest same table
        col_diff = abs(col_count1 - col_count2) if col_count1 and col_count2 else 0
        
        return col_diff <= self.max_column_difference
    
    def get_column_count(self, cells: List[TableCell]) -> int:
        """Get the maximum column count for a list of table cells"""
        if not cells:
            return 0
        
        max_cols = None
        for row_id in set([cell.row_id for cell in cells]):
            row_cells = [cell for cell in cells if cell.row_id == row_id]
            cols = 0
            for cell in row_cells:
                cols += cell.colspan
            if max_cols is None or cols > max_cols:
                max_cols = cols
        return max_cols or 0

    def process_table_pair(self, document: Document, table1: Block, table2: Block):
        """Process a pair of tables for potential corrections"""
        print(f"\n=== DEBUG: Analyzing table pair for corrections ===")
        print(f"Table 1: {table1.id} on page {table1.page_id}")
        print(f"Table 2: {table2.id} on page {table2.page_id}")
        
        cells1 = table1.contained_blocks(document, (BlockTypes.TableCell,))
        cells2 = table2.contained_blocks(document, (BlockTypes.TableCell,))
        
        if not cells1 or not cells2:
            print("DEBUG: Skipping - missing cells")
            return
        
        # Get images
        try:
            image1 = table1.get_image(document, highres=False)
            image2 = table2.get_image(document, highres=False)
            print(f"DEBUG: Images extracted - table1: {image1.size}, table2: {image2.size}")
        except Exception as e:
            print(f"DEBUG: Error getting images: {e}")
            return
        
        # Get HTML representations
        try:
            html1 = self.render_html_with_row_ids(table1, document)
            html2 = self.render_html_with_row_ids(table2, document)
            print(f"DEBUG: HTML rendered - table1: {len(html1)} chars, table2: {len(html2)} chars")
        except Exception as e:
            print(f"DEBUG: Error rendering HTML: {e}")
            return
        
        # Create prompt
        prompt = self.correction_prompt.replace("{{table1_html}}", html1).replace("{{table2_html}}", html2)
        # Call LLM
        try:
            print("DEBUG: Calling LLM for correction analysis...")
            response = self.llm_service(
                prompt,
                [image1, image2],
                table1,
                CorrectionSchema,
            )
            print(f"DEBUG: LLM response: {response}")
        except Exception as e:
            print(f"DEBUG: LLM service error: {e}")
            return
        
        if not response or "corrections" not in response:
            print("DEBUG: Invalid response format")
            return
        
        # Apply corrections
        corrections_data = response.get("corrections", [])
        if corrections_data:
            print(f"DEBUG: Applying {len(corrections_data)} corrections...")
            # Convert dict responses to schema objects
            corrections = []
            for correction_dict in corrections_data:
                try:
                    correction = CorrectionItemSchema(**correction_dict)
                    corrections.append(correction)
                except Exception as e:
                    print(f"DEBUG: Error parsing correction {correction_dict}: {e}")
                    continue
            
            if corrections:
                self.apply_corrections(cells1, cells2, corrections)
        else:
            print("DEBUG: No corrections needed")

    def apply_corrections(self, cells1: List[TableCell], cells2: List[TableCell], corrections: List[CorrectionItemSchema]):
        """Apply the specified corrections to the table cells"""
        for correction in corrections:
            try:
                self.apply_single_correction(cells1, cells2, correction)
            except Exception as e:
                print(f"DEBUG: Error applying correction {correction}: {e}")
    
    def apply_single_correction(self, cells1: List[TableCell], cells2: List[TableCell], correction: CorrectionItemSchema):
        """Apply a single correction operation"""
        correction_type = correction.type
        source_loc = correction.source
        
        # Get source cell
        source_cells = cells1 if source_loc.table_index == 0 else cells2
        source_cell = self.find_cell_at_location(source_cells, source_loc.row, source_loc.col)
        
        if not source_cell:
            print(f"DEBUG: Could not find source cell at {source_loc}")
            return
        
        if correction_type == "merge_cells":
            target_loc = correction.target
            target_cells = cells1 if target_loc.table_index == 0 else cells2
            target_cell = self.find_cell_at_location(target_cells, target_loc.row, target_loc.col)
            
            if not target_cell:
                print(f"DEBUG: Could not find target cell at {target_loc}")
                return
            
            action = correction.action
            source_text = " ".join(source_cell.text_lines) if source_cell.text_lines else ""
            
            if action == "append":
                if target_cell.text_lines:
                    target_cell.text_lines[-1] += " " + source_text
                else:
                    target_cell.text_lines = [source_text]
            elif action == "prepend":
                if target_cell.text_lines:
                    target_cell.text_lines[0] = source_text + " " + target_cell.text_lines[0]
                else:
                    target_cell.text_lines = [source_text]
            elif action == "replace":
                target_cell.text_lines = [source_text]
            
            # Clear the source cell
            source_cell.text_lines = []
            
            print(f"DEBUG: Merged '{source_text}' into target cell using action '{action}'")
        
        elif correction_type == "move_cell":
            # Similar to merge_cells but replaces target content
            target_loc = correction.target
            target_cells = cells1 if target_loc.table_index == 0 else cells2
            target_cell = self.find_cell_at_location(target_cells, target_loc.row, target_loc.col)
            
            if target_cell:
                target_cell.text_lines = source_cell.text_lines.copy()
                source_cell.text_lines = []
                print(f"DEBUG: Moved cell content from {source_loc} to {target_loc}")
        
        elif correction_type == "remove_cell":
            source_cell.text_lines = []
            print(f"DEBUG: Removed cell content at {source_loc}")
        
        elif correction_type == "move_text_fragment":
            target_loc = correction.target
            text_fragment = correction.text_fragment
            
            if not target_loc or not text_fragment:
                print(f"DEBUG: move_text_fragment requires target and text_fragment")
                return
            
            target_cells = cells1 if target_loc.table_index == 0 else cells2
            target_cell = self.find_cell_at_location(target_cells, target_loc.row, target_loc.col)
            
            if not target_cell:
                print(f"DEBUG: Could not find target cell at {target_loc}")
                return
            # Find and remove the text fragment from source cell
            source_text = " ".join(source_cell.text_lines) if source_cell.text_lines else ""
            if text_fragment not in source_text:
                print(f"DEBUG: Text fragment '{text_fragment}' not found in source cell")
                return
            
            # Remove the fragment from source (with some cleanup)
            remaining_text = source_text.replace(text_fragment, "").strip()
            # Clean up multiple spaces and line breaks
            remaining_text = " ".join(remaining_text.split())
            source_cell.text_lines = [remaining_text] if remaining_text else []
            
            # Add the fragment to target cell
            action = correction.action
            if action == "append":
                if target_cell.text_lines:
                    target_cell.text_lines[-1] += " " + text_fragment
                else:
                    target_cell.text_lines = [text_fragment]
            elif action == "prepend":
                if target_cell.text_lines:
                    target_cell.text_lines[0] = text_fragment + " " + target_cell.text_lines[0]
                else:
                    target_cell.text_lines = [text_fragment]
            elif action == "replace":
                target_cell.text_lines = [text_fragment]
            
            print(f"DEBUG: Moved text fragment '{text_fragment}' from {source_loc} to {target_loc} using action '{action}'")
    
    def find_cell_at_location(self, cells: List[TableCell], row: int, col: int) -> TableCell:
        """Find a cell at the specified row/column location"""
        # Handle negative indexing for row
        if row < 0:
            max_row = max(cell.row_id for cell in cells) if cells else -1
            row = max_row + row + 1
        
        print(f"DEBUG: Looking for cell at row {row}, col {col}")
        
        # Create a simple lookup dictionary
        cell_lookup = {(cell.row_id, cell.col_id): cell for cell in cells}
        
        if (row, col) in cell_lookup:
            found_cell = cell_lookup[(row, col)]
            print(f"DEBUG: Found cell: {' '.join(found_cell.text_lines or [])}")
            return found_cell
        
        print(f"DEBUG: No cell found at row {row}, col {col}")
        print(f"DEBUG: Available cells: {list(cell_lookup.keys())}")
        return None
    
    def find_cell_by_content(self, cells: List[TableCell], content_pattern: str, col: int) -> TableCell:
        """Find a cell by content pattern in the specified column"""
        for cell in cells:
            if cell.col_id == col:
                cell_text = ' '.join(cell.text_lines or [])
                if content_pattern.lower() in cell_text.lower():
                    print(f"DEBUG: Found cell by content '{content_pattern}' at row {cell.row_id}: {cell_text}")
                    return cell
        return None

    def render_html_with_row_ids(self, table: Block, document: Document) -> str:
        """Render table HTML with row IDs added to each <tr> element"""
        html = json_to_html(table.render(document))
        
        # Add row IDs to each <tr> element
        row_counter = 0
        
        def replace_tr(match):
            nonlocal row_counter
            result = f'<tr id="row-{row_counter}">'
            row_counter += 1
            return result
        
        # Use regex to replace all <tr> tags with numbered ones
        import re
        modified_html = re.sub(r'<tr>', replace_tr, html)
        
        return modified_html
    
    def find_better_target_cell(self, cells: List[TableCell], current_target: TableCell, text_fragment: str, col: int) -> TableCell:
        """Find a better target cell if the current one seems inappropriate"""
        current_content = ' '.join(current_target.text_lines or []).lower()
        
        # Skip if current target looks like a case number or has lots of text (likely complete)
        if any(pattern in current_content for pattern in ['case#', 'tax:', 'action', 'return']):
            print(f"DEBUG: Current target looks like a case/tax entry, searching for incomplete product...")
            
            # Look for cells that seem incomplete (shorter, product-like names)
            candidates = []
            for cell in cells:
                if cell.col_id == col and cell != current_target:
                    cell_text = ' '.join(cell.text_lines or [])
                    cell_text_lower = cell_text.lower()
                    
                    # Look for product-like entries that seem incomplete
                    if (any(keyword in cell_text_lower for keyword in ['osv', 'platform', 'app', 'software']) and
                        not any(pattern in cell_text_lower for pattern in ['case#', 'tax:', 'action', 'return']) and
                        len(cell_text) < len(current_content)):  # Prefer shorter (potentially incomplete) entries
                        candidates.append((cell, len(cell_text)))
            
            # Return the shortest candidate (most likely to be incomplete)
            if candidates:
                best_candidate = min(candidates, key=lambda x: x[1])[0]
                print(f"DEBUG: Found better candidate: {' '.join(best_candidate.text_lines or [])}")
                return best_candidate
        
        return current_target