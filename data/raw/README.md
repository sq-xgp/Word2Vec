# Raw data

The local corpus is Leipzig Corpora Collection `eng_wikipedia_2016_10K`.

- Source page: <https://wortschatz.uni-leipzig.de/en/download/eng>
- Download archive: `eng_wikipedia_2016_10K.tar.gz`
- Archive SHA-256: `d0c803b7b10d7b42e2da0a3990ded945143594be89f135266da0e8998ffe4edc`
- Local sentence file: `eng_wikipedia_2016_10K-sentences.txt`
- Format: UTF-8, one `Sentence_ID<TAB>Sentence` record per line
- Terms: <https://www.wortschatz.uni-leipzig.de/en/usage>

The archive and extracted corpus are ignored by Git. Download them from the source instead of
committing third-party data to the repository.

The larger semantic-quality experiment uses `eng_wikipedia_2016_100K` from the same source.

- Download archive: `eng_wikipedia_2016_100K.tar.gz`
- Archive SHA-256: `04aa301072a612e0368f1a0abe5f6b011ab03df84961c29b80bd12683a5a6f0`
- Sentence file: `eng_wikipedia_2016_100K-sentences.txt`
- Size: 100,000 sentences and 2,121,562 cleaned tokens
