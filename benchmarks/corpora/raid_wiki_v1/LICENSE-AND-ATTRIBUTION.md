# Licensing and attribution for `raid_wiki_v1`

This directory contains third-party text. **It is not covered by this
repository's MIT license**, which applies to the source code in `server/`,
`benchmarks/*.py`, `scripts/`, and `tests/`. Nothing in this directory is part
of the distributed `writing-tools-mcp` package.

## The dataset

Every document here was extracted from the **RAID** dataset, `train` split,
Wikipedia (`wiki`) domain, non-adversarial (`attack == "none"`) rows only.

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

RAID's declared MIT license covers the RAID authors' own contribution - the
generations, the labels, and the dataset's assembly. It does not, and cannot,
relicense the human-written source text.

The human-labeled documents in this corpus are **excerpts of English Wikipedia
articles**, which Wikipedia licenses under
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) (and, for older
revisions, additionally under the GFDL). Every human document therefore carries
attribution and share-alike obligations independent of RAID's MIT grant:

- **Attribution.** Each human record's `source.title` field holds the Wikipedia
  article title it was taken from, so any individual document can be traced back
  to its article. The corpus as a whole is attributed to Wikipedia contributors
  via <https://en.wikipedia.org>.
- **Share-alike.** Redistributing these documents, or an adapted version of
  them, carries the CC BY-SA 4.0 requirement to license the redistribution under
  the same terms. That obligation attaches to this corpus directory, not to the
  MIT-licensed code that reads it.

Where the two statements differ, treat the stricter one as governing the human
documents: MIT for the RAID assembly and the machine generations, CC BY-SA 4.0
for the Wikipedia prose. RAID does not record the exact article revision each
excerpt came from, so revision-level attribution is not available and article
titles are the finest granularity this corpus can offer.
