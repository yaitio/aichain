"""Code blocks and tables longer than the limit: the packer's paths, pinned."""

from yait_aichain.tools.chunk import chunk


def _cut(text):
    return [(d["text"], d["metadata"]["is_code"]) for d in chunk(max_chars=40, overlap_chars=0).run(text)]


def test_a_long_code_block_repeats_its_fences():
    code = "```py\n" + "\n".join(f"x = {i}" for i in range(8)) + "\n```"
    assert _cut(code) == [("```py\nx = 0\nx = 1\nx = 2\nx = 3\nx = 4\n```", True),
                          ("```py\nx = 5\nx = 6\nx = 7\n```", True)]


def test_a_code_line_over_the_budget_is_cut_by_characters():
    assert _cut("```\n" + "y" * 50 + "\nz\n```") == [
        ("```\n" + "y" * 32 + "\n```", True), ("```\n" + "y" * 18 + "\n```", True), ("```\nz\n```", True)]


def test_a_long_table_repeats_its_header():
    table = "| a | b |\n|---|---|\n" + "\n".join(f"| {i} | {i * i} |" for i in range(8))
    head = "| a | b |\n|---|---|\n"
    assert _cut(table) == [(head + "| 0 | 0 |\n| 1 | 1 |", False), (head + "| 2 | 4 |\n| 3 | 9 |", False),
                           (head + "| 4 | 16 |", False), (head + "| 5 | 25 |", False),
                           (head + "| 6 | 36 |", False), (head + "| 7 | 49 |", False)]


def test_a_table_row_over_the_budget_is_cut_by_characters():
    assert _cut("| a |\n|---|\n| " + "r" * 40 + " |\n| s |") == [
        ("| a |\n|---|\n| " + "r" * 26, False), ("| a |\n|---|\n" + "r" * 14 + " |", False),
        ("| a |\n|---|\n| s |", False)]


def test_a_header_with_no_room_for_a_row_is_split_as_text():
    assert _cut("| " + "h" * 40 + " |\n|---|\n| 1 |") == [
        ("|", False), ("h" * 40, False), (" |\n|---|\n| 1 |", False)]


def test_text_code_and_a_table_in_one_section():
    text = ("# T\nintro text\n```\nq = 1\nw = 2\ne = 3\nr = 4\n```\n"
            "| k | v |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |\n| 5 | 6 |\nafter")
    assert _cut(text) == [("intro text", False), ("```\nq = 1\nw = 2\ne = 3\nr = 4\n```", True),
                          ("| k | v |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |", False),
                          ("| k | v |\n|---|---|\n| 5 | 6 |\nafter", False)]
