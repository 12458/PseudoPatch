# PseudoPatch

A pure-Python utility for applying human-readable “pseudo-diff” patch files to a collection of text files.

This library is designed for situations where the full complexity of standard `diff` and `patch` tools is unnecessary, and a simpler, more readable patch format is preferred. It is self-contained and requires no external dependencies.

## Features

  * **Pure Python:** Runs anywhere Python 3.9+ is installed. No other dependencies required.
  * **Simple, Readable Format:** Uses a human-readable format for patch files.
  * **Core Git-like Actions:** Supports adding, deleting, and updating files.
  * **File Renaming:** Supports moving/renaming files as part of an update.
  * **Flexible API:** Can be used as a command-line tool or as a library in your own Python projects.
  * **Error Handling:** Provides clear error messages for malformed patches or failed applications.

## Installation

You can install the library directly from PyPI:

```bash
pip install pseudopatch
```

Or, if you have a local copy, you can install it using:

```bash
pip install .
```

## Usage

PseudoPatch can be used as a command-line tool or integrated into your Python code.

### 1\. Command-Line Interface (CLI)

The CLI is designed to be simple and work with standard shell pipelines. It reads the patch content from `stdin`.

**Example:**

Assuming you have a patch file named `my_changes.patch`, you can apply it to your local files like this:

```bash
cat my_changes.patch | pseudopatch
```

Or using input redirection:

```bash
pseudopatch < my_changes.patch
```

The tool will automatically read the necessary files, apply the changes, and write the results back to the filesystem.

### 2\. Library API

You can import `pseudopatch` to integrate its functionality directly into your application. This gives you full control over file I/O.

The main entry point is the `pseudopatch.api.process_patch` function. It requires you to provide your own functions for reading, writing, and removing files.

**Example:**

```python
import os
from pseudopatch import api
from pseudopatch.exceptions import DiffError

# --- 1. Define your patch text ---
patch_text = """
*** Begin Patch ***
*** Update File: src/main.py ***
*** Move to: src/app.py ***
@@ import sys
-from utils import old_helper
+from helpers import new_helper

 def main():
-    old_helper()
+    new_helper()
     print("Done.")
*** Add File: src/helpers.py ***
+def new_helper():
+    print("This is the new helper function.")
*** Delete File: src/utils.py ***
*** End Patch ***
"""

# --- 2. Define your file system handlers ---
# You have full control over how files are read and written.
# This could interact with a database, a virtual filesystem, etc.

def read_from_disk(path: str) -> str:
    """Reads a file's content."""
    print(f"-> Reading '{path}'")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def write_to_disk(path: str, content: str) -> None:
    """Writes content to a file, creating directories if needed."""
    print(f"<- Writing to '{path}'")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

def remove_from_disk(path: str) -> None:
    """Removes a file."""
    print(f"xx Removing '{path}'")
    try:
        os.remove(path)
    except FileNotFoundError:
        pass # It's okay if the file is already gone

# --- 3. Process the patch ---
try:
    print("Applying patch...")
    result = api.process_patch(
        text=patch_text,
        open_fn=read_from_disk,
        write_fn=write_to_disk,
        remove_fn=remove_from_disk,
    )
    print(f"\nSuccess: {result}")

except DiffError as e:
    print(f"\nAn error occurred: {e}", file=os.sys.stderr)

```

## Patch File Syntax

The patch file format is designed to be easy to read and write by hand.

### Full Example

```
*** Begin Patch ***
# Comments can be added on lines that don't start with '***', '+', '-', or ' '.

*** Update File: old_name.txt ***
*** Move to: new_name.txt ***
@@ This is a context line that helps locate the change.
-This line will be deleted.
+This line will be added in its place.
+This is another line that will be added.
 This line remains unchanged and provides context.
-This line is also removed.

*** Add File: new_file.log ***
+This is the first line of the new file.
+This is the second line.

*** Delete File: obsolete.dat ***

*** End Patch ***
```

### Syntax Breakdown

#### General Structure

  * Every patch must start with `*** Begin Patch ***` and end with `*** End Patch ***`.

#### File Actions

A patch consists of one or more file actions.

  * **Update a File:**

    ```
    *** Update File: path/to/your/file.txt ***
    ```

    This section describes changes within an existing file. It can optionally be followed by a move/rename instruction:

    ```
    *** Move to: path/to/new_name.txt ***
    ```

  * **Add a File:**

    ```
    *** Add File: path/to/new/file.txt ***
    ```

    All subsequent lines starting with `+` will form the content of this new file.

  * **Delete a File:**

    ```
    *** Delete File: path/to/unwanted/file.txt ***
    ```

    This action has no body and simply marks the file for deletion.

#### Content Lines (for Updates and Adds)

  * `+This is an added line.`

      * Lines prefixed with `+` are added to the file.

  * `-This is a deleted line.`

      * Lines prefixed with `-` are removed from the file. This is only valid in an `*** Update File ***` block.

  * `  This is a context line. `

      * Lines prefixed with a single space are "context" lines. They are not changed but are used to locate the position of the patch. They are only valid in an `*** Update File ***` block.

  * `@@ Optional context anchor`

      * A line starting with `@@` can be used to "anchor" the subsequent patch chunk to a specific location in the source file, improving accuracy if the file has many similar-looking lines.

## Reference Prompt

```python
APPLY_PATCH_TOOL_DESC = """This is a custom utility that makes it more convenient to add, remove, move, or edit code files. `apply_patch` effectively allows you to execute a diff/patch against a file, but the format of the diff specification is unique to this task, so pay careful attention to these instructions. To use the `apply_patch` command, you should pass a message of the following structure as "input":

%%bash
apply_patch <<"EOF"
*** Begin Patch
[YOUR_PATCH]
*** End Patch
EOF

Where [YOUR_PATCH] is the actual content of your patch, specified in the following V4A diff format.

*** [ACTION] File: [path/to/file] -> ACTION can be one of Add, Update, or Delete.
For each snippet of code that needs to be changed, repeat the following:
[context_before] -> See below for further instructions on context.
- [old_code] -> Precede the old code with a minus sign.
+ [new_code] -> Precede the new, replacement code with a plus sign.
[context_after] -> See below for further instructions on context.

For instructions on [context_before] and [context_after]:
- By default, show 3 lines of code immediately above and 3 lines immediately below each change. If a change is within 3 lines of a previous change, do NOT duplicate the first change’s [context_after] lines in the second change’s [context_before] lines.
- If 3 lines of context is insufficient to uniquely identify the snippet of code within the file, use the @@ operator to indicate the class or function to which the snippet belongs. For instance, we might have:
@@ class BaseClass
[3 lines of pre-context]
- [old_code]
+ [new_code]
[3 lines of post-context]

- If a code block is repeated so many times in a class or function such that even a single @@ statement and 3 lines of context cannot uniquely identify the snippet of code, you can use multiple `@@` statements to jump to the right context. For instance:

@@ class BaseClass
@@ 	def method():
[3 lines of pre-context]
- [old_code]
+ [new_code]
[3 lines of post-context]

Note, then, that we do not use line numbers in this diff format, as the context is enough to uniquely identify code. An example of a message that you might pass as "input" to this function, in order to apply a patch, is shown below.

%%bash
apply_patch <<"EOF"
*** Begin Patch
*** Update File: pygorithm/searching/binary_search.py
@@ class BaseClass
@@     def search():
-          pass
+          raise NotImplementedError()

@@ class Subclass
@@     def search():
-          pass
+          raise NotImplementedError()

*** End Patch
EOF
"""

APPLY_PATCH_TOOL = {
    "name": "apply_patch",
    "description": APPLY_PATCH_TOOL_DESC,
    "parameters": {
        "type": "object",
        "properties": {
            "input": {
                "type": "string",
                "description": " The apply_patch command that you wish to execute.",
            }
        },
        "required": ["input"],
    },
}
```

## Contributing

Contributions are welcome\! If you find a bug or have a feature request, please open an issue.

To contribute code:

1.  Fork the repository.
2.  Create a new branch for your feature (`git checkout -b feature/my-new-feature`).
3.  Install the development dependencies: `pip install -e .`
4.  Add your changes and commit them (`git commit -am 'Add some feature'`).
5.  Push to the branch (`git push origin feature/my-new-feature`).
6.  Create a new Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.