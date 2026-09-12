<!-- g:tool -->
# Local files and shell

## `local_browse` — `LocalBrowseTool`

List directory contents as a tree.

| | |
|---|---|
| Import | `from yait_aichain.tools import LocalBrowseTool` |
| Risk class | `write` |

```python
LocalBrowseTool(
    root_dir: str | None = None,
)
```

| Name | Type | Required | Description |
|---|---|---|---|
| `input` | `string` | ✓ | Directory path to list. Relative to sandbox root. Default: root itself. |
| `options.depth` | `integer` |  | How many directory levels to show (default 2). |
| `options.pattern` | `string` |  | Glob pattern to filter files, e.g. '*.py'. |
| `options.show_hidden` | `boolean` |  | Include hidden files and directories (default false). |

```python
result = LocalBrowseTool()(
    input="…",
    options={"depth": …},
)
```

## `local_read` — `LocalReadTool`

Read a file's contents.

| | |
|---|---|
| Import | `from yait_aichain.tools import LocalReadTool` |
| Risk class | `write` |

```python
LocalReadTool(
    root_dir: str | None = None,
)
```

| Name | Type | Required | Description |
|---|---|---|---|
| `input` | `string` | ✓ | Path to the file to read (relative to sandbox root). |
| `options.start_line` | `integer` |  | First line to return, 1-indexed (default 1). |
| `options.end_line` | `integer` |  | Last line to return, inclusive (default: all). |
| `options.encoding` | `string` |  | File encoding (default utf-8). |

```python
result = LocalReadTool()(
    input="…",
    options={"start_line": …},
)
```

## `local_write` — `LocalWriteTool`

Write content to a file.

| | |
|---|---|
| Import | `from yait_aichain.tools import LocalWriteTool` |
| Risk class | `write` |

```python
LocalWriteTool(
    root_dir: str | None = None,
)
```

| Name | Type | Required | Description |
|---|---|---|---|
| `input` | `string` | ✓ | Content to write to the file. |
| `options.path` | `string` | ✓ | Destination file path (relative to sandbox root). |
| `options.mode` | `string` |  | 'w' to overwrite (default), 'a' to append. One of `w`, `a`. |
| `options.encoding` | `string` |  | File encoding (default utf-8). |

```python
result = LocalWriteTool()(
    input="…",
    options={"path": …},
)
```

## `local_run` — `LocalRunTool`

Execute a shell command or Python file.

| | |
|---|---|
| Import | `from yait_aichain.tools import LocalRunTool` |
| Risk class | `write` |

```python
LocalRunTool(
    root_dir: str | None = None,
)
```

| Name | Type | Required | Description |
|---|---|---|---|
| `input` | `string` | ✓ | Shell command to execute, e.g. 'ls -la' or 'python script.py'. If a .py file path is given, it runs with the current Python. |
| `options.timeout` | `integer` |  | Max seconds before the process is killed (default 30). |
| `options.cwd` | `string` |  | Working directory (relative to sandbox root, default: root). |
| `options.python` | `boolean` |  | Force execution with the current Python interpreter. |

```python
result = LocalRunTool()(
    input="…",
    options={"timeout": …},
)
```
<!-- /g:tool -->
