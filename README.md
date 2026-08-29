# Writing Tools MCP Server

This is a Model Context Protocol (MCP) server designed to provide various text analysis tools, assisting users in improving their writing. It is **optimized for Claude Desktop** with one-click installation via MCP bundles, and also works with other MCP-compatible tools like Cursor and Windsurf.

MCP servers act as a secure bridge or interface, enabling AI models and language assistants to interact with local applications, tools, or data on a user's machine. This server leverages that protocol to offer its specialized writing-specific analysis capabilities to connected AI clients.

## Features

This server provides the following text analysis tools:

*   **`list_tools`**: List all available tools in this server.
*   **`character_count`**: Return the number of characters in the input text.
*   **`word_count`**: Return the number of words in the input text.
*   **`spellcheck`**: Return a list of misspelled words in the input text.
*   **`readability_score`**: Return readability scores (Flesch, Kincaid, Fog) for the text, section, or paragraph level.
*   **`reading_time`**: Return the estimated reading time for the text, section, or paragraph level.
*   **`keyword_density`**: Calculate the density of a given keyword in the text.
*   **`keyword_frequency`**: Count how often each keyword appears in the text (optionally removing stopwords).
*   **`top_keywords`**: Identify the most frequently used keywords in the text.
*   **`keyword_context`**: Extract sentences or phrases where a specific keyword appears.
*   **`passive_voice_detection`**: Detect passive voice constructions in the text.
*   **`perplexity_analysis`**: Analyze text for perplexity and burstiness to detect AI-generated content using GPT-2.
*   **`stylometric_analysis`**: Analyze stylometric features (sentence length, length-robust lexical diversity, POS ratios, six readability grade-level formulas, syntactic complexity, per-function-word/Burrows' Delta profile) for AI detection, against a built-in or custom baseline.

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
  custom_baselines_dir: "data/baselines/custom_baselines"
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
`data/baselines/custom_baselines/` and is immediately usable:

```
stylometric_analysis(text, baseline="my_own_voice")
```

By default the builder (`server.stylometry.build_baseline_from_texts`) only computes mean/std for
a curated, length-robust feature set: `avg_sentence_len`, `sentence_len_std`, `fog`/`kincaid`/`smog`/
`coleman_liau`/`ari`/`dale_chall` (six readability grade-level formulas), `mtld`/`mattr` (length-robust
lexical diversity), `mean_dependency_distance`/`subordinate_clause_ratio` (syntactic complexity read
off the dependency parse), the `ADP`/`DET` POS ratios, and a Burrows'-Delta-style per-function-word
frequency profile (see below). Type-token ratio and the hapax legomena rate are deliberately left
out: both fall monotonically as a document gets longer, for any author, so comparing them across a
corpus of mixed document lengths mostly measures length rather than style - `mtld` and `mattr` exist
specifically as length-robust replacements for them (McCarthy & Jarvis 2010; Covington & McFall 2010).
`fourgram_repetition_rate` and `zipf_slope` are computed but *not* in the default set: unlike
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

`data/baselines/custom_baselines/mcginniscommawill_pre2020.json` ships as a worked example: 102
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
*   **Returns**: `float` - The density percentage ( (keyword count / total words) * 100 ).

---

**`keyword_frequency`**

*   **Description**: Count how often each keyword (token) appears in the text.
*   **Parameters**:
    *   `text` (`str`): The text to analyze.
    *   `remove_stopwords` (`bool`, optional, default=`True`): Whether to exclude common English stopwords (e.g., 'the', 'a', 'is').
*   **Returns**: `dict` - A dictionary mapping each keyword (or lemma) to its frequency count.

---

**`top_keywords`**

*   **Description**: Identify the most frequently used keywords in the text.
*   **Parameters**:
    *   `text` (`str`): The text to analyze.
    *   `top_n` (`int`, optional, default=`10`): The number of top keywords to return.
    *   `remove_stopwords` (`bool`, optional, default=`True`): Whether to exclude common English stopwords.
*   **Returns**: `list[tuple[str, int]]` - A list of tuples, where each tuple contains a keyword (or lemma) and its count, sorted by frequency in descending order.

---

**`keyword_context`**

*   **Description**: Extract sentences where a specific keyword (case-insensitive, lemmatized) appears. Multi-word keywords are matched as complete, contiguous phrases.
*   **Parameters**:
    *   `text` (`str`): The text to search within.
    *   `keyword` (`str`): The keyword or phrase to find.
*   **Returns**: `list[str]` - A list of sentences containing the keyword or phrase, matched on lemmas.

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

*   **Description**: Analyze text for stylometric features and detect AI-generated content. Computes sentence length distribution, lexical diversity (TTR/Hapax, plus the length-robust `mtld`/`mattr`), POS ratios, six readability grade-level formulas (Fog, Kincaid, SMOG, Coleman-Liau, ARI, Dale-Chall), syntactic complexity from the dependency parse (`mean_dependency_distance`, `subordinate_clause_ratio`), n-gram repetition and Zipf-slope, and a per-function-word frequency profile reduced to a Burrows' Delta score. Flags outliers relative to a baseline (built-in `brown_corpus`, or a [custom baseline](#custom-baselines) built from your own writing) using z-score analysis.
*   **Parameters**:
    *   `text` (`str`): The text to analyze.
    *   `baseline` (`str`, optional, default=`"brown_corpus"`): Baseline corpus name for comparison. See [Custom Baselines](#custom-baselines) to build your own.
    *   `language` (`str`, optional, default=`"en"`): Language code (only "en" supported currently).
*   **Returns**: `dict` - Stylometric analysis including:
    *   `features` (`dict`): Extracted stylometric features (sentence length, TTR/hapax/`mtld`/`mattr`, POS ratios, `fog`/`kincaid`/`smog`/`coleman_liau`/`ari`/`dale_chall`, `mean_dependency_distance`, `subordinate_clause_ratio`, `fourgram_repetition_rate`, `zipf_slope`, `function_word_freqs`, etc.)
    *   `z_scores` (`dict`): Z-scores of features against the baseline, including per-word `fw_<word>` scores and the aggregate `burrows_delta`
    *   `flags` (`dict`): AI detection flags with confidence levels and explanations
    *   `sentence_analysis` (`list`): Per-sentence analysis with z-scores
    *   `config` (`dict`): Baseline information and analysis thresholds

---

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

This is MIT licensed
