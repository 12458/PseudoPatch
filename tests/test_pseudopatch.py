import pytest
from pathlib import Path
import tempfile
import os
from typing import Dict, List, Callable

from pseudopatch.domain import ActionType, Chunk, Commit, FileChange, Patch, PatchAction
from pseudopatch.exceptions import DiffError
from pseudopatch.patching import Parser, patch_to_commit, _get_updated_file, find_context
from pseudopatch.api import (
    text_to_patch,
    identify_files_needed,
    identify_files_added,
    load_files,
    apply_commit,
    process_patch,
)


class TestParsing:
    def test_identify_files_needed(self):
        patch_text = """*** Begin Patch
*** Update File: file1.txt
*** Delete File: file2.txt
*** Add File: file3.txt
*** End Patch"""
        assert identify_files_needed(patch_text) == ["file1.txt", "file2.txt"]

    def test_identify_files_added(self):
        patch_text = """*** Begin Patch
*** Update File: file1.txt
*** Delete File: file2.txt
*** Add File: file3.txt
*** End Patch"""
        assert identify_files_added(patch_text) == ["file3.txt"]

    def test_text_to_patch_basic(self):
        patch_text = """*** Begin Patch
*** Update File: file1.txt
@@ First line
 Line 1
-Line 2
+Line 2 modified
 Line 3
*** End Patch"""
        orig_files = {"file1.txt": "First line\nLine 1\nLine 2\nLine 3"}
        patch, fuzz = text_to_patch(patch_text, orig_files)

        assert len(patch.actions) == 1
        assert "file1.txt" in patch.actions
        assert patch.actions["file1.txt"].type == ActionType.UPDATE
        assert len(patch.actions["file1.txt"].chunks) == 1
        assert patch.actions["file1.txt"].chunks[0].del_lines == ["Line 2"]
        assert patch.actions["file1.txt"].chunks[0].ins_lines == ["Line 2 modified"]

    def test_text_to_patch_invalid_format(self):
        patch_text = "Not a valid patch"
        with pytest.raises(DiffError, match="Invalid patch text - missing sentinels"):
            text_to_patch(patch_text, {})

    def test_text_to_patch_missing_file(self):
        patch_text = """*** Begin Patch
*** Update File: missing.txt
*** End Patch"""
        with pytest.raises(DiffError, match="Update File Error - missing file"):
            text_to_patch(patch_text, {})


class TestPatching:
    def test_get_updated_file(self):
        text = "line1\nline2\nline3\nline4"
        action = PatchAction(type=ActionType.UPDATE)

        # Replace line2 with new_line2
        chunk = Chunk(orig_index=1, del_lines=["line2"], ins_lines=["new_line2"])
        action.chunks.append(chunk)

        result = _get_updated_file(text, action, "test.txt")
        assert result == "line1\nnew_line2\nline3\nline4"

    def test_get_updated_file_multiple_chunks(self):
        text = "line1\nline2\nline3\nline4\nline5"
        action = PatchAction(type=ActionType.UPDATE)

        # Replace line2 with new_line2
        chunk1 = Chunk(orig_index=1, del_lines=["line2"], ins_lines=["new_line2"])
        action.chunks.append(chunk1)

        # Replace line4 with new_line4
        chunk2 = Chunk(orig_index=3, del_lines=["line4"], ins_lines=["new_line4"])
        action.chunks.append(chunk2)

        result = _get_updated_file(text, action, "test.txt")
        assert result == "line1\nnew_line2\nline3\nnew_line4\nline5"

    def test_get_updated_file_out_of_bounds(self):
        text = "line1\nline2"
        action = PatchAction(type=ActionType.UPDATE)

        # Try to modify at an invalid index
        chunk = Chunk(orig_index=5, del_lines=["nonexistent"], ins_lines=["new"])
        action.chunks.append(chunk)

        with pytest.raises(DiffError, match="chunk.orig_index .* exceeds file length"):
            _get_updated_file(text, action, "test.txt")

    def test_get_updated_file_overlapping_chunks(self):
        text = "line1\nline2\nline3"
        action = PatchAction(type=ActionType.UPDATE)

        # Two overlapping chunks
        chunk1 = Chunk(orig_index=1, del_lines=["line2"], ins_lines=["new_line2"])
        chunk2 = Chunk(orig_index=0, del_lines=["line1"], ins_lines=["new_line1"])
        action.chunks.append(chunk1)
        action.chunks.append(chunk2)

        with pytest.raises(DiffError, match="overlapping chunks"):
            _get_updated_file(text, action, "test.txt")

    def test_patch_to_commit(self):
        patch = Patch()
        # Add file action
        patch.actions["new.txt"] = PatchAction(
            type=ActionType.ADD,
            new_file="This is a new file"
        )

        # Update file action
        update_action = PatchAction(type=ActionType.UPDATE)
        chunk = Chunk(orig_index=0, del_lines=["old"], ins_lines=["new"])
        update_action.chunks.append(chunk)
        patch.actions["update.txt"] = update_action

        # Delete file action
        patch.actions["delete.txt"] = PatchAction(type=ActionType.DELETE)

        orig = {
            "update.txt": "old\nsecond line",
            "delete.txt": "content to delete"
        }

        commit = patch_to_commit(patch, orig)

        assert len(commit.changes) == 3
        assert commit.changes["new.txt"].type == ActionType.ADD
        assert commit.changes["new.txt"].new_content == "This is a new file"

        assert commit.changes["update.txt"].type == ActionType.UPDATE
        assert commit.changes["update.txt"].old_content == "old\nsecond line"
        assert commit.changes["update.txt"].new_content == "new\nsecond line"

        assert commit.changes["delete.txt"].type == ActionType.DELETE
        assert commit.changes["delete.txt"].old_content == "content to delete"

    def test_find_context(self):
        lines = ["line1", "line2", "line3", "line4"]
        context = ["line2", "line3"]

        # Exact match
        idx, fuzz = find_context(lines, context, 0, False)
        assert idx == 1
        assert fuzz == 0

        # No match
        idx, fuzz = find_context(lines, ["nonexistent"], 0, False)
        assert idx == -1

        # Fuzzy match with whitespace differences
        lines_with_spaces = ["line1", "  line2  ", "line3", "line4"]
        idx, fuzz = find_context(lines_with_spaces, context, 0, False)
        assert idx == 1
        assert fuzz > 0


class TestFileOperations:
    def setup_method(self):
        # Create a temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        self.base_path = Path(self.temp_dir.name)

    def teardown_method(self):
        # Clean up the temporary directory
        self.temp_dir.cleanup()

    def write_file(self, filename: str, content: str):
        path = self.base_path / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            f.write(content)
        return str(path)

    def read_file(self, filename: str) -> str:
        path = self.base_path / filename
        with path.open("r") as f:
            return f.read()

    def test_load_files(self):
        self.write_file("file1.txt", "content1")
        self.write_file("file2.txt", "content2")

        paths = [str(self.base_path / "file1.txt"), str(self.base_path / "file2.txt")]
        files = load_files(paths, lambda p: open(p, 'r').read())

        assert len(files) == 2
        assert files[str(self.base_path / "file1.txt")] == "content1"
        assert files[str(self.base_path / "file2.txt")] == "content2"

    def test_apply_commit(self):
        # Set up test files
        file1_path = self.write_file("existing.txt", "original content")
        file2_path = str(self.base_path / "new.txt")

        # Create a commit
        commit = Commit()

        # Update existing file
        commit.changes[file1_path] = FileChange(
            type=ActionType.UPDATE,
            old_content="original content",
            new_content="updated content"
        )

        # Add new file
        commit.changes[file2_path] = FileChange(
            type=ActionType.ADD,
            new_content="new file content"
        )

        # Apply the commit
        def write_fn(path, content):
            with open(path, 'w') as f:
                f.write(content)

        def remove_fn(path):
            if os.path.exists(path):
                os.remove(path)

        apply_commit(commit, write_fn, remove_fn)

        # Check results
        assert self.read_file("existing.txt") == "updated content"
        assert self.read_file("new.txt") == "new file content"

    def test_process_patch(self):
        # Create test files
        file1_path = self.write_file("file1.txt", "line1\nline2\nline3")

        # Create patch text
        patch_text = f"""*** Begin Patch
*** Update File: {file1_path}
@@ line1
 line1
-line2
+line2 modified
 line3
*** Add File: {self.base_path}/file2.txt
+new file content
*** End Patch"""

        # Process the patch
        def open_fn(path):
            with open(path, 'r') as f:
                return f.read()

        def write_fn(path, content):
            with open(path, 'w') as f:
                f.write(content)

        def remove_fn(path):
            if os.path.exists(path):
                os.remove(path)

        result = process_patch(patch_text, open_fn, write_fn, remove_fn)

        # Check results
        assert result == "Done!"
        assert self.read_file("file1.txt") == "line1\nline2 modified\nline3"
        assert self.read_file("file2.txt") == "new file content"


class TestErrorHandling:
    def test_invalid_patch_start(self):
        patch_text = "Invalid patch"
        with pytest.raises(DiffError, match="Patch text must start with"):
            process_patch(patch_text, lambda _: "", lambda _a, _b: None, lambda _: None)

    def test_duplicate_actions(self):
        patch_text = """*** Begin Patch
*** Update File: file.txt
*** Update File: file.txt
*** End Patch"""
        with pytest.raises(DiffError, match="Duplicate update for file"):
            text_to_patch(patch_text, {"file.txt": "line1"})

    def test_add_existing_file(self):
        patch_text = """*** Begin Patch
*** Add File: file.txt
+content
*** End Patch"""
        with pytest.raises(DiffError, match="Add File Error - file already exists"):
            text_to_patch(patch_text, {"file.txt": "existing content"})


if __name__ == "__main__":
    pytest.main([__file__])
