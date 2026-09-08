# Writing Tools MCP Server

This is a Model Context Protocol (MCP) server designed to provide various text analysis tools, assisting users in improving their writing. It is **optimized for Claude Desktop** with one-click installation via MCP bundles, and also works with other MCP-compatible tools like Cursor and Windsurf.

MCP servers act as a secure bridge or interface, enabling AI models and language assistants to interact with local applications, tools, or data on a user's machine. This server leverages that protocol to offer its specialized writing-specific analysis capabilities to connected AI clients.

## Features

This server provides the following text analysis tools:

*   **`list_tools`**: List all available tools in this server.
*   **`character_count`**: Return the number of characters in the input text.
*   **`word_count`**: Return the number of words in the input text.
*   **`spellcheck`**: Return a list of misspelled words in the input text.
*   **`readability_score`**: Return readability scores (Flesch, Kincaid, Fog) for the text, section, or paragraph level. Successful responses include a `findings` array.
*   **`reading_time`**: Return the estimated reading time for the text, section, or paragraph level.
*   **`keyword_density`**: Calculate the density of a given keyword in the text. Returns `{"keyword", "density", "findings"}`.
*   **`keyword_frequency`**: Count how often each keyword appears in the text (optionally removing stopwords). Returns `{"frequencies", "findings"}`.
*   **`top_keywords`**: Identify the most frequently used keywords in the text. Returns `{"keywords", "findings"}`.
*   **`keyword_context`**: Extract sentences or phrases where a specific keyword appears. Returns `{"keyword", "sentences", "findings"}`.
*   **`passive_voice_detection`**: Detect passive voice constructions in the text.
*   **`perplexity_analysis`**: Analyze text for perplexity and burstiness to detect AI-generated content using GPT-2. Successful responses include a `findings` array.
*   **`stylometric_analysis`**: Analyze stylometric features (sentence length, length-robust lexical diversity and vocabulary rarity, POS ratios and bigrams, six readability grade-level formulas, syntactic complexity, punctuation idiosyncrasies, hedge/booster rate, per-function-word/Burrows' Delta profile, character n-gram profile) for AI detection, against a built-in or custom baseline. Successful responses include a `findings` array.

## Install

```bash
# Run directly from GitHub (no install needed)
uvx --from git+https://github.com/wdm0006/writing-tools-mcp writing-tools-mcp

# Or install from source
git clone https://github.com/wdm0006/writing-tools-mcp
cd writing-tools-mcp
uv sync
uv run run_server.py
```

The spaCy `en_core_web_sm` model is not published to PyPI (Explosion distributes
model wheels through [GitHub releases](https://github.com/explosion/spacy-models/releases)),
so it is not listed as a package dependency. It is downloaded automatically on
first use; to pre-install it - for example to keep test runs hermetic - run:

```bash
uv pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
```

## MCP Client Configuration

```json
{
  "mcpServers": {
    "writingtools": {
      "command": "uvx",
      "args": ["--from", "git+https://github.com/wdm0006/writing-tools-mcp", "writing-tools-mcp"]
    }
  }
}
```

## Server Configuration

The server reads an optional `.mcp-config.yaml` from its working directory. Unknown keys and
wrongly typed values are ignored with a warning on stderr, and every missing key falls back to
the default below.

```yaml
perplexity:
  model_name: "gpt2"     # Hugging Face model used for perplexity analysis
  max_length: 512        # Token window per chunk
  overlap: 50            # Token overlap between chunks
  device: "cpu"          # "cpu" pins the model to CPU
  language: "en"         # Only "en" is supported
  thresholds:
    ppl_max: 25.0        # Perplexity at or below this counts as an AI signal
    burstiness_min: 2.5  # Burstiness below this counts as an AI signal

stylometry:
  default_baseline: "brown_corpus"
  custom_baselines_dir: "server/data/baselines/custom_baselines"
  thresholds:
    warning_z: 2.0                # |z| for a warning
    error_z: 3.0                  # |z| for an error
    ai_confidence_threshold: 0.7  # Confidence needed to flag AI authorship
  features:
    enabled: ["sentence_length", "ttr", "hapax", "pos_ratios", "punctuation", "function_words"]
    pos_tags: ["NOUN", "VERB", "ADJ", "ADV", "ADP", "DET", "PRON", "CONJ", "NUM", "PART"]

logging:
  level: "INFO"    # CRITICAL, ERROR, WARNING, INFO, or DEBUG
  format: "%(asctime)s - %(levelname)s - %(message)s"  # Standard `logging` format string
```

Set `logging.level: "DEBUG"` when reporting a problem. Logs are always written to stderr — stdout
carries the MCP JSON-RPC stream — and an unrecognized level falls back to `INFO` with a warning
rather than stopping the server.

`perplexity.language`, `stylometry.default_baseline`, `stylometry.custom_baselines_dir` and
`stylometry.features` are accepted and type-checked, but nothing reads them yet; the baseline and
language are chosen per call through the `stylometric_analysis` and `perplexity_analysis`
arguments.

## Custom Baselines

`stylometric_analysis` compares a text's features against a baseline (`brown_corpus` by default)
and flags whatever is an outlier relative to it. Brown Corpus is 1961 published news and fiction;
it answers "does this read like typical published prose," which is often not the question you
actually want answered. A more useful question for judging your own drafts is "does this read like
*my own* pre-existing writing" - answered by building a baseline from a corpus of your own text.

```bash
uv run scripts/build_baseline.py my_own_voice path/to/txt/files/
```

Each `*.txt` file in the directory is treated as one document (strip front matter, markdown, and
code fences first - the script analyzes exactly the text it's given). The baseline is saved under
`server/data/baselines/custom_baselines/` (inside the `server` package, so it is found whether the
server runs from a checkout or an installed wheel) and is immediately usable:

```
stylometric_analysis(text, baseline="my_own_voice")
```

By default the builder (`server.stylometry.build_baseline_from_texts`) only computes mean/std for
a curated, length-robust feature set: `avg_sentence_len`, `sentence_len_std`, `fog`/`kincaid`/`smog`/
`coleman_liau`/`ari`/`dale_chall` (six readability grade-level formulas), `mtld`/`mattr`/`mtld_lemma`
(length-robust lexical diversity, on surface forms and lemmas respectively), `mean_word_frequency`
(vocabulary *rarity*, via `wordfreq` - distinct from diversity: how common the words used are, not
how many distinct words there are), `word_len_std`, `lexical_density`, five punctuation-idiosyncrasy
ratios (`semicolon_ratio`, `em_dash_ratio`, `ellipsis_ratio`, `exclamation_ratio`,
`parenthetical_rate`), `hedge_rate`/`booster_rate` (epistemic-marker word categories),
`mean_dependency_distance`/`subordinate_clause_ratio` (syntactic complexity read off the dependency
parse), the `ADP`/`DET` POS ratios, a curated 10-bigram POS-sequence profile (see below), a
Burrows'-Delta-style per-function-word frequency profile (see below), and a character n-gram
orthographic profile (see below). Type-token ratio and the hapax legomena rate are deliberately left
out: both fall monotonically as a document gets longer, for any author, so comparing them across a
corpus of mixed document lengths mostly measures length rather than style - `mtld`/`mattr`/`mtld_lemma`
exist specifically as length-robust replacements for them (McCarthy & Jarvis 2010; Covington & McFall
2010). `fourgram_repetition_rate` and `zipf_slope` are computed but *not* in the default set: unlike
ttr/hapax, we haven't verified whether they vary with length, so they're opt-in only. Pass
`--all-features` to include every feature `extract_features` computes, length-confounded or not.

**Burrows' Delta.** Rather than one aggregate `function_word_ratio`, the builder also tracks each of
`StylemetricAnalyzer`'s ~100 function words individually (mean/std per word across the corpus).
`stylometric_analysis` z-scores each word against its own baseline entry, then reduces all of them
to one number - `burrows_delta`, the mean absolute z-score across every word scored - the classic
Burrows' Delta statistic (Burrows 2002), built for exactly this kind of small, single-author corpus.
A large `burrows_delta` (above the usual warning z-threshold) raises a `distinct_function_word_profile`
flag. Pass `function_words=[]` to `build_baseline_from_texts` (or a custom word list) to change or
skip this dimension.

**POS bigrams.** Published authorship-attribution work reports POS-tag bigrams/trigrams
discriminating authors substantially better than single-tag POS ratios alone. The builder tracks a
small, curated 10-bigram subset by default (`DEFAULT_ROBUST_POS_BIGRAMS` - noun- and
verb-phrase-initiation patterns like `DET_NOUN`, `VERB_ADP`), scored the same way as `pos_ratios`
under a `posbi_` prefix, rather than all ~289 possible tag combinations - most bigrams are too sparse
per document (a handful of occurrences in an 800-word post) to average reliably. Pass `pos_bigrams=`
to change the tracked set, or `[]` to skip this dimension.

**Character n-gram profile.** A PAN/CLEF-style orthographic fingerprint: character 4-gram relative
frequencies, normalized per document. Individual n-grams are too sparse to z-score the way pos_ratios
or function words are (most 4-grams occur 0-2 times in a typical post), so this is compared as a
*whole profile* instead - `calculate_char_ngram_similarity` computes the cosine similarity between a
draft's profile and the baseline's aggregate profile (kept to the top `char_ngram_top_k` n-grams by
corpus-wide frequency, default 300, to bound the baseline's file size). `stylometric_analysis` surfaces
this as a top-level `char_ngram_similarity` (not part of `z_scores` or `flags` - there's no calibrated
threshold for it yet). Note this is sensitive to vocabulary/topic, not just style: a post using very
different subject-matter vocabulary from the baseline corpus will score a low similarity for that
reason alone, not necessarily because of authorship. Pass `char_ngram_top_k=0` to skip this dimension.

`server/data/baselines/custom_baselines/mcginniscommawill_pre2020.json` ships as a worked example: 102
pre-2020 posts from [mcginniscommawill.com](https://mcginniscommawill.com), built with this script.

## Building the Bundle

To create a `.mcpb` bundle for distribution:

```bash
make build-mcpb
```

This creates `writing-tools-mcp.mcpb` which can be installed in Claude Desktop.

## Usage Examples

You can configure any MCP client (like Claude.ai, Windsurf, or Cursor) to connect to it. Here are some example prompts you could give to an AI assistant connected to this MCP server:

**General Analysis:**

*   "List the available writing tools." (Calls `list_tools`)
*   "Analyze the text below for readability using the standard scores." (Provide text, calls `readability_score`)
*   "Check this document for spelling mistakes." (Provide text, calls `spellcheck`)
*   "How long would it take someone to read this blog post?" (Provide text, calls `reading_time`)

**Keyword Analysis:**

*   "What are the top 5 keywords in the following abstract?" (Provide text, calls `top_keywords` with `top_n=5`)
*   "Calculate the keyword density for 'artificial intelligence' in this paper." (Provide text, calls `keyword_density` with `keyword="artificial intelligence"`)
*   "Show me all sentences containing the term 'MCP'." (Provide text, calls `keyword_context` with `keyword="MCP"`)
*   "Search the web for pages based on the top 5 keyworkds in this text, and compare those pages to mine" (Provide text, calls `top_keywords` with `top_n=5`, then passes that to a different web search tool if available)

**Style and Structure:**

*   "Identify any sentences using passive voice in my draft." (Provide text, calls `passive_voice_detection`)
*   "What's the word count for this paragraph?" (Provide text, calls `word_count`)
*   "Get the readability scores for each section of this document." (Provide markdown text, calls `readability_score` with `level="section"`)

**AI Detection:**

*   "Analyze this text for signs of AI generation using perplexity analysis." (Provide text, calls `perplexity_analysis`)
*   "Check if this essay was written by AI using stylometric analysis." (Provide text, calls `stylometric_analysis`)
*   "Compare the writing style of this text against human writing baselines." (Provide text, calls `stylometric_analysis`)
*   "Is this text too uniform in sentence structure to be human-written?" (Provide text, calls both AI detection tools)

## Revision Prompts

Two MCP prompts support an analyze → revise → verify loop:

*   **`guided_revision`**: Render an impact-ordered revision brief for a document. Pass the `findings` array from any analysis tool as the optional `findings` argument (JSON string); every finding's `rule`, `location`, `message`, and `fix_hint` is listed, highest-impact first. Omit it and the brief tells you which tools to run first.
*   **`writing_checklist`**: A pre-flight drafting checklist (structure, sentence variety, hedging and boosters, readability, keywords, voice) to apply while writing.

Seven analysis tools (`readability_score`, `perplexity_analysis`, `stylometric_analysis`, `keyword_density`, `keyword_frequency`, `top_keywords`, `keyword_context`) attach a `findings` array to successful responses — located, actionable observations with `rule`, `location`, `message`, and `fix_hint` fields, where fix hints coach the fix rather than restate the flaw. Stylometric findings are honestly scoped to the chosen baseline: indicators the baseline cannot measure produce no finding. Error responses are unchanged, and `passive_voice_detection` still returns a plain list of sentences.

## Tool Reference

Below is a detailed reference for each tool provided by the server.

---

**`list_tools`**

*   **Description**: List all available tools in this server.
*   **Parameters**: None
*   **Returns**: `list[str]` - A list of tool names.

---

**`character_count`**

*   **Description**: Return the number of characters in the input text.
*   **Parameters**:
    *   `text` (`str`): The input text.
*   **Returns**: `int` - The total character count.

---

**`word_count`**

*   **Description**: Return the number of words in the input text.
*   **Parameters**:
    *   `text` (`str`): The input text.
*   **Returns**: `int` - The total word count (based on whitespace splitting).

---

**`spellcheck`**

*   **Description**: Return a list of misspelled words in the input text.
*   **Parameters**:
    *   `text` (`str`): The input text.
*   **Returns**: `list[str]` - A list of words identified as potentially misspelled.

---

**`readability_score`**

*   **Description**: Return readability scores using Flesch Reading Ease, Flesch-Kincaid Grade Level, and Gunning Fog index.
*   **Parameters**:
    *   `text` (`str`): The text to analyze.
    *   `level` (`str`, optional): Granularity of analysis. Options:
        *   `"full"` (default): Score the entire text.
        *   `"section"`: Score the full text and each markdown section (identified by `#` headings) separately.
        *   `"paragraph"`: Score the full text and each paragraph (separated by blank lines) separately.
*   **Returns**: `dict` - A dictionary containing the scores. Structure depends on the `level` parameter. For `"full"`, it returns `{"flesch": float, "kincaid": float, "fog": float}`. For other levels, it returns nested dictionaries. Returns `None` for scores if the text segment is too short.

---

**`reading_time`**

*   **Description**: Return the estimated reading time for the input text (based on `textstat`). Markdown markup is stripped before estimating, so syntax characters and link/image target URLs do not count toward the time.
*   **Parameters**:
    *   `text` (`str`): The text to analyze.
    *   `level` (`str`, optional): Granularity of analysis. Options:
        *   `"full"` (default): Calculate for the entire text.
        *   `"section"`: Calculate for the full text and each markdown section.
        *   `"paragraph"`: Calculate for the full text and each paragraph.
*   **Returns**: `dict` - A dictionary containing the estimated reading time in minutes. Structure depends on the `level` parameter.

---

**`keyword_density`**

*   **Description**: Calculate the density of a given keyword in the text (case-insensitive, lemmatized). Multi-word keywords are matched as complete, contiguous phrases.
*   **Parameters**:
    *   `text` (`str`): The text to analyze.
    *   `keyword` (`str`): The keyword or phrase to search for.
*   **Returns**: `dict` - `{"keyword": str, "density": float, "findings": list}` — the density percentage ( (keyword count / total words) * 100 ) plus actionable findings.

---

**`keyword_frequency`**

*   **Description**: Count how often each keyword (token) appears in the text.
*   **Parameters**:
    *   `text` (`str`): The text to analyze.
    *   `remove_stopwords` (`bool`, optional, default=`True`): Whether to exclude common English stopwords (e.g., 'the', 'a', 'is').
*   **Returns**: `dict` - `{"frequencies": {keyword: count, ...}, "findings": list}` — the frequency map (counts under `frequencies`) plus actionable findings.

---

**`top_keywords`**

*   **Description**: Identify the most frequently used keywords in the text.
*   **Parameters**:
    *   `text` (`str`): The text to analyze.
    *   `top_n` (`int`, optional, default=`10`): The number of top keywords to return.
    *   `remove_stopwords` (`bool`, optional, default=`True`): Whether to exclude common English stopwords.
*   **Returns**: `dict` - `{"keywords": [[keyword, count], ...], "findings": list}` — keyword/count pairs sorted by frequency in descending order, plus actionable findings.

---

**`keyword_context`**

*   **Description**: Extract sentences where a specific keyword (case-insensitive, lemmatized) appears. Multi-word keywords are matched as complete, contiguous phrases.
*   **Parameters**:
    *   `text` (`str`): The text to search within.
    *   `keyword` (`str`): The keyword or phrase to find.
*   **Returns**: `dict` - `{"keyword": str, "sentences": list[str], "findings": list}` — the matching sentences plus actionable findings.

---

**`passive_voice_detection`**

*   **Description**: Detect sentences containing passive voice constructions (based on a simplified pattern matching using spaCy).
*   **Parameters**:
    *   `text` (`str`): The text to analyze.
*   **Returns**: `list[str]` - A list of sentences identified as potentially containing passive voice.

---

**`perplexity_analysis`**

*   **Description**: Analyze text for perplexity and burstiness to detect AI-generated content using GPT-2. Computes document-level and sentence-level perplexity along with burstiness (variance of perplexity across sentences). Low perplexity combined with low burstiness is a statistical signal used by AI detectors.
*   **Parameters**:
    *   `text` (`str`): The text to analyze.
    *   `language` (`str`, optional, default=`"en"`): Language code (only "en" supported currently).
*   **Returns**: `dict` - Analysis results including:
    *   `doc_ppl` (`float | null`): Document-level perplexity score; `null` when no sentence could be scored
    *   `doc_burstiness` (`float | null`): Burstiness score (standard deviation of sentence perplexities); `null` when fewer than two sentences were scored, since the standard deviation is undefined there
    *   `sentences` (`list`): Sentence-level perplexity scores
    *   `config` (`dict`): Model configuration and thresholds
    *   `flags` (`dict`): AI detection flags with confidence and explanations

---

**`stylometric_analysis`**

*   **Description**: Analyze text for stylometric features and detect AI-generated content. Computes sentence length distribution, lexical diversity (TTR/Hapax, plus the length-robust `mtld`/`mattr`/`mtld_lemma`) and vocabulary rarity (`mean_word_frequency`, via `wordfreq`), POS ratios and a curated POS-bigram profile, six readability grade-level formulas (Fog, Kincaid, SMOG, Coleman-Liau, ARI, Dale-Chall), syntactic complexity from the dependency parse (`mean_dependency_distance`, `subordinate_clause_ratio`), punctuation idiosyncrasies (semicolon/em-dash/ellipsis/exclamation/parenthetical rate), hedge/booster epistemic-marker rates, n-gram repetition and Zipf-slope, a per-function-word frequency profile reduced to a Burrows' Delta score, and a character n-gram orthographic profile compared via cosine similarity. Flags outliers relative to a baseline (built-in `brown_corpus`, or a [custom baseline](#custom-baselines) built from your own writing) using z-score analysis.
*   **Parameters**:
    *   `text` (`str`): The text to analyze.
    *   `baseline` (`str`, optional, default=`"brown_corpus"`): Baseline corpus name for comparison. See [Custom Baselines](#custom-baselines) to build your own.
    *   `language` (`str`, optional, default=`"en"`): Language code (only "en" supported currently).
*   **Returns**: `dict` - Stylometric analysis including:
    *   `features` (`dict`): Extracted stylometric features (sentence length, TTR/hapax/`mtld`/`mattr`/`mtld_lemma`, `mean_word_frequency`, `word_len_std`, `lexical_density`, POS ratios, `pos_bigram_ratios`, `fog`/`kincaid`/`smog`/`coleman_liau`/`ari`/`dale_chall`, `mean_dependency_distance`, `subordinate_clause_ratio`, punctuation-idiosyncrasy ratios, `hedge_rate`/`booster_rate`, `fourgram_repetition_rate`, `zipf_slope`, `function_word_freqs`, etc. - `char_ngram_profile` is computed internally for `char_ngram_similarity` below but omitted here, as a several-hundred-entry intermediate)
    *   `z_scores` (`dict`): Z-scores of features against the baseline, including per-word `fw_<word>` scores and the aggregate `burrows_delta`, and per-bigram `posbi_<tag>_<tag>` scores
    *   `flags` (`dict`): AI detection flags with confidence levels and explanations
    *   `sentence_analysis` (`list`): Per-sentence analysis with z-scores
    *   `char_ngram_similarity` (`float | null`): Cosine similarity between this text's character n-gram profile and the baseline's (see [Custom Baselines](#custom-baselines)); `null` when the baseline has no character n-gram profile (e.g. `brown_corpus`)
    *   `config` (`dict`): Baseline information and analysis thresholds

---

## Detector Benchmark

`benchmarks/` is an offline evaluation surface, separate from the MCP server. It scores
the AI-detection code against committed labeled corpora and writes a report, so the
shipped thresholds can be argued about with numbers instead of intuition. It changes no
tool, default, or threshold.

```bash
uv run benchmarks/run_benchmark.py
```

See [`benchmarks/README.md`](benchmarks/README.md) for what the corpora are, what the
report contains, and how far the results reproduce. Corpus text is third-party and is
licensed separately from this repository's code.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

This is MIT licensed
