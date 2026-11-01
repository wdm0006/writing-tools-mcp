# Licensing and attribution for `raid_abstracts_v1`

This directory contains third-party text. **It is not covered by this
repository's MIT license**, which applies to the source code in `server/`,
`benchmarks/*.py`, `scripts/`, and `tests/`. Nothing in this directory is part
of the distributed `writing-tools-mcp` package.

## The dataset

Every document here was extracted from the **RAID** dataset, `train` split,
abstracts (`abstracts`) domain, non-adversarial (`attack == "none"`) rows only.

- Dataset: <https://huggingface.co/datasets/liamdugan/raid>
- Project: <https://github.com/liamdugan/raid> and <https://raid-bench.xyz>
- Revision pinned for this extraction: `865cac74188466cb0c3b7574a10204007b57a459`

RAID is distributed under the MIT License. Its license text, reproduced from
<https://github.com/liamdugan/raid/blob/main/LICENSE>:

```
MIT License

Copyright (c) 2024- Liam Dugan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

Please cite the RAID paper when using or reporting on this corpus:

```bibtex
@inproceedings{dugan-etal-2024-raid,
    title = "{RAID}: A Shared Benchmark for Robust Evaluation of Machine-Generated Text Detectors",
    author = "Dugan, Liam and Hwang, Alyssa and Trhl{\'\i}k, Filip and Zhu, Andrew and
      Ludan, Josh Magnus and Xu, Hainiu and Ippolito, Daphne and Callison-Burch, Chris",
    booktitle = "Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.acl-long.674",
    pages = "12463--12492",
}
```

## The upstream text underneath

RAID's MIT license covers the RAID authors' own contribution - the generations,
the labels, and the dataset's assembly. It does not relicense the
human-written source text.

The human-labeled documents in this corpus are **abstracts of arXiv papers**
(RAID's `abstracts` domain is built from the arXiv metadata snapshot; see the
RAID paper's dataset description). Licensing conclusion:

- **Redistributable.** arXiv states that the metadata it exposes, which
  includes each paper's title and abstract, is released under
  [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/)
  (<https://info.arxiv.org/help/bulk_data.html>). CC0 imposes no attribution or
  share-alike obligation, so the human documents are committed verbatim.
- **Scope of that conclusion.** It rests on arXiv's statement about its
  metadata and on RAID's stated provenance for the domain. The abstract text is
  copyright its authors; the CC0 grant comes from arXiv's metadata terms, not
  from per-paper licenses, and this repository has not audited individual
  papers. If either upstream statement changes, remove this directory and
  rebuild it locally with the command in `MANIFEST.md`.
- **Attribution anyway.** Each human record's `source.title` is the paper
  title, so any document can be traced to its arXiv entry. Please cite RAID
  (above) and acknowledge arXiv when reporting results.
- **Machine documents** are RAID generations, covered by RAID's MIT license.
  The model providers' terms of use were not separately reviewed.
