# Cookbooks

Finished programs on aichain, one task each that a person actually has. Each
directory is the whole program: copied out on its own, with its
`requirements.txt` installed, it runs. Each has an answer key that needs no
person (`python run.py --check`) and prints what its run cost.

| Recipe | What it does | Needs |
|---|---|---|
| [cite](cite/README.md) | answers from three EU regulations and a scanned page, with the article that says so | Anthropic, Google; MarkItDown; Chroma |
| [desk](desk/README.md) | a support agent that refunds small, asks a manager for big, and runs as two serverless handlers | Anthropic |
| [council](council/README.md) | models from different vendors vote, rank each other or argue to agreement — and whether agreement means truth | any three model keys |
| [reconcile](reconcile/README.md) | invoices against a contract and delivery notes, four agent strategies on one answer key | Anthropic |
| [survey](survey/README.md) | a cited brief on a company from its 10-K filings and the web, with every disagreement listed | Anthropic, Google, a search key; MarkItDown; Chroma |
| [recall](recall/README.md) | a long conversation remembered by byheart and questioned in a fresh process, scored LoCoMo's way | Anthropic; byheart |
| [brief](brief/README.md) | a ten-page literature brief on twelve papers, written a section at a time | Anthropic; MarkItDown |

## On every aichain release

Each recipe is checked in its own directory — there is no shared runner, as
there is no shared code:

```bash
for d in cite desk council reconcile survey recall brief; do (cd $d && python run.py --check); done
```

`council` has no `run.py`: its scripts are run one by one, and `truth.py` is
its measured part. `reconcile` is `reconcile.py --cases 3`. `cite` and `survey`
need Chroma running (`chroma run --path data/chroma`); `survey` needs
`SEC_USER_AGENT`.
