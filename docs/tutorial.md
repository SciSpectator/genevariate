# Tutorial

A first session, from an empty window to an exported region. It uses a GEO
platform downloaded through the program itself, so every number on screen comes
from public data you can fetch again.

Installation is in [INSTALL.md](../INSTALL.md). This page assumes `genevariate`
starts.

## 0. Get a platform

Open **Tools > Download Platform...**, pick a GEO platform and let the
configurator download and normalize it. The registry record, the series it
lists and the ingestion route are shown before anything is fetched.

## 1. Load a platform

Start the program.

```bash
genevariate
```

The downloaded platform appears in the list on the
left with its sample count.

The filename matters. The program reads the `GPLxxxxx` out of it to pair the
expression file with its label file, so keep the accession in the name. The
`series_id` column in the file is preserved and skips the GEOmetadb lookup,
which is what makes the study-clumping corrections work offline.

For real data the other three buttons are the ones you want:

- a preset button, or **Download Platform** for any other GPL
- **Load RNA-seq from ARCHS4** for uniformly reprocessed bulk counts
- **Single-cell (CELLxGENE)** for a Census query, aggregated to per-donor
  per-cell-type profiles on the way in

**Discover Experiments** searches GEOmetadb, ARCHS4, CELLxGENE and Expression
Atlas by keyword if you do not already know the accession.

## 2. Attach labels

Press the label button for the platform and pick
a label file of your own.

Open **Label Entities**. On real data this window is where you find out what
your labels actually are: which values resolved to a MeSH heading, which
resolved to a Cellosaurus cell line, and which were only clustered locally
because no vocabulary recognised them. A `CVCL_` accession under Tissue means
the sample is a catalogued cell line grown in a flask, not a biopsy, and
counting it as a tissue is a claim the extractor never made.

A label file without accessions will show every value as
unresolved. That is correct.

If you have no label file, **Extract Labels** runs the vendored
[LLM-GEO-Label-Extractor](https://github.com/SciSpectator/LLM-GEO-Label-Extractor)
against an OpenAI-compatible endpoint you point it at. Only five fields are
produced: Tissue, Condition, Treatment, Sex, Age.

## 3. Brush a region

Open **Gene Distribution Explorer**, type `GENE_A` and plot it. You get one
panel per selected platform.

Drag across the right-hand tail. The selection turns into a shaded band and the
sample count updates as you drag. Scroll to zoom, right-drag to pan,
double-click to reset. Clicking a single sample opens its record: the study it
came from, the five labels, its free-text characteristics and its expression
values.

Add `GENE_B` and brush its tail too. Two brushed genes make a box, and the
samples inside it are the ones satisfying both.

Press **Analyze Selected Range**.

## 4. Read the region

The region window opens with eight tabs. Every one of them takes the same region
and the same label column, which you choose in the left panel.

Set the label column to **Tissue** and work left to right.

**Labels** and **Frequency**. Liver should dominate the A-and-B box. That is the
planted rule.

**Enrichment**. Liver near the top with a small q-value. Look at the two columns
that are not in a normal enrichment table:

- `n_GSE`, how many studies contributed
- `n_eff`, the sample count after correcting for the fact that they came in
  studies

Now switch the label column to a region containing **Blood**. Blood was planted
in four studies and is driven by no gene at all. Its raw count looks
respectable and its `n_eff` collapses, because rho is near 1. That is the point
of the column. A large count from few studies is one experiment repeated, and
the interface says so on the row rather than in a footnote.

Switch the label column to **Condition**, which is pure noise. After
Benjamini-Hochberg nothing should survive. If something does, you have found a
bug and the ground-truth document is the thing to quote.

**Comparison**. Brush a second region, on another gene or another platform, and
this tab tests the regions against each other rather than each against the
background. Both regions can be enriched against the platform while being
indistinguishable from each other, which is a result the enrichment tab cannot
express. Overlapping regions share samples, so the Jaccard overlap is carried on
every pairwise row.

**Statistics**, **Samples**. The numeric summary and the sample rows themselves.

## 5. Export

**Export All** at the bottom of the window writes every tab, including the ones
that only draw when you press a button. Those are computed for you during the
export rather than being silently left out.

You also get a Save button, a right-click menu and Ctrl+S on every single table
in the program, and a Save button on every plot, with PNG and PDF written side
by side. File names are documented in [output-schema.md](output-schema.md).

## 6. Across platforms

Load a second platform and label it, then use **Cross-Platform Analysis** for
whole-platform comparison and the region window's Comparison tab for regions.

Across platforms the region window **pools** rather than concatenating. Each
platform is a stratum with its own background and the effect sizes are combined
by DerSimonian-Laird random effects, so a large platform cannot impose its label
mix on a small one. Cochran's Q and I^2 tell you whether the platforms agree.

The one thing you have to get right is the region definition. Technologies are
related at best by an unknown monotone transform, so brush by quantile. "The top
decile of this gene on this platform" transfers between technologies. "Above 12"
does not.

## 7. The assistant

**Ctrl+/** opens an assistant that drives the same analysis API the buttons do.

```text
load GPL570
analyse the distribution of ALB on GPL570
which tissues are enriched in the top decile of ALB
compare ALB across microarray and rna-seq modalities
run meta enrichment across GPL570 and GPL96 tumor vs normal
```

It needs an OpenAI-compatible endpoint. Any local server such as vLLM, SGLang or
Ollama will serve it.

```bash
export GENEVARIATE_AGENT_BACKEND=ollama
export OLLAMA_URL=http://127.0.0.1:11434
export GENEVARIATE_AGENT_MODEL=<model name>
```

`ollama` is the default backend. For `groq` set `GROQ_API_KEY` instead, and for
any other hosted endpoint set `GENEVARIATE_AGENT_BASE_URL` and
`GENEVARIATE_AGENT_API_KEY`.

It is given the computed numbers and asked to explain them, never to produce
them. Results are written into the assistant's own window so the chat stays
readable, and every answer carries a manifest of which tool ran with which
arguments, so you can reproduce it by hand.

## Where to go next

- [Architecture and data flow](architecture.md)
- [Methods and theory](methods.md)
- [Output schema](output-schema.md)
- [Reproducibility and limitations](reproducibility.md)
