# Model card: Medicine Leaflet Assistant

## Intended use

Answering factual questions about what a medicine's **package leaflet** says (use, warnings, how to take it,
side effects, storage, contents), with a citation to the leaflet section.

**Not intended for:** diagnosis, treatment decisions, dosing decisions for a specific person, emergencies,
or any use as a medical device. It is a portfolio project.

## System

* Retrieval over chunks of public package leaflets (TF-IDF by default; optional sentence embeddings).
* Generation by an OpenAI-compatible LLM restricted to the retrieved excerpts, or extractive sentence
  selection without an LLM.
* Rule-based guardrails before retrieval; a retrieval-score threshold for "not found".

## Data

* Source: public package leaflets (e.g. EMA product information). Only the patient leaflet part is indexed.
* No user data is used for training. No model is trained or fine-tuned.
* Tests and CI use three fictional leaflets that are clearly marked as fictional.

## Evaluation

See the README for the metrics and how to reproduce them. Evaluation covers retrieval, answer content,
citations, abstention on off-topic questions, refusal of emergencies and personal-advice questions,
and false refusals on normal questions.

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Hallucinated medical facts | Prompt restricted to excerpts; citations required; `NOT_IN_SOURCES` abstention; temperature 0; extractive mode quotes text verbatim |
| Mixing information from different medicines | Answers use passages from one leaflet only |
| User relies on the answer for a personal decision | Personal-advice questions are refused and referred to a pharmacist or doctor; every answer carries a disclaimer |
| Emergency not recognised | Emergency patterns return an emergency-number message before anything else; tested on a fixed set. Paraphrases may be missed (see Limitations) |
| Outdated leaflet | Leaflets are versioned files; the index must be rebuilt when they change |
| Questions are personal health data | Logs store a hash of the question, not the text; no question storage; LLM provider choice (incl. local Ollama) lets data stay on premises |

## Limitations

* Keyword-based guardrails miss paraphrases; a production system needs a trained classifier and monitoring of refusals.
* English only.
* Automatic metrics do not replace manual review of a sample of answers.
