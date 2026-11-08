# Preprint submission handoff

Prepared 7 November 2026. Status: local draft; not submitted. Preprint-first route selected by the author. arXiv is the proposed destination; `cs.CV` is a candidate category, subject to account and moderation requirements.

## Files and fields

- Paper: [object_attention_paper.pdf](../pdf/object_attention_paper.pdf). Submit the single paper PDF, not this handoff or the entire repository.
- Title: **Robustness of Object Attention Prediction With Annotated and Detected Objects**.
- Sole author: **Göktuğ Sina Bekçioğulları**, spelling confirmed by the author.
- ORCID: [0009-0001-2026-1019](https://orcid.org/0009-0001-2026-1019), supplied by the author. Stored in submission metadata, not added to the existing PDF author block.
- Affiliation: **Department of Computer Engineering, Marmara University, İstanbul, Türkiye**. This records current student affiliation; university funding, supervision or endorsement is not claimed.
- Abstract: copy the `abstract` field from [metadata.json](metadata.json), which is checked against the manuscript.
- Comments: **7 pages, 1 figure, 3 tables. Exploratory OSIE and public EMOd study.**
- Journal reference and publication DOI: leave empty; there is no accepted journal article.
- Distribution license: explicitly left undecided by the author for now.
- arXiv account: the author reports they have not registered yet.
- Evidence review: [publication_review.md](../../research/publication_review.md).

## Submission sequence

1. Review the complete PDF and confirm the title and distribution license. The sole-author spelling and current student affiliation have been recorded. Preserve the assistance acknowledgment.
2. Register the author's arXiv account and complete the account's verification steps, then sign in. Check the account's eligibility and any endorsement required for the chosen category. Associate the supplied ORCID through the account's supported flow if desired. Start with the [official registration guidance](https://info.arxiv.org/help/registerhelp.html).
3. Upload the single PDF produced from the Word manuscript. Enter the matching title, author and abstract from metadata. Do not describe IEEE formatting as IEEE acceptance.
4. Inspect arXiv's processed preview and any processing messages. Local font/text checks do not replace this step.
5. Confirm the final submission and later record the assigned arXiv identifier. Moderation and announcement are separate from preparing a local draft.

## Platform guidance checked on 7 November 2026

arXiv accepts PDF as a submission format and expects authors to self-submit; submissions undergo moderation. See the [submission overview](https://info.arxiv.org/help/submit/index.html).

For this Word-authored manuscript, the [PDF instructions](https://info.arxiv.org/help/submit_pdf.html) call for a single PDF with embedded outline fonts and machine-readable text. Existing local checks report eight embedded fonts and extractable body text. Equations are raster illustrations, so those checks are not a full accessibility certification.

First-time/category eligibility must be checked for the actual account; the [endorsement guidance](https://info.arxiv.org/help/endorsement.html) explains the process. Supplying an ORCID does not complete it.

The author must choose the [distribution license](https://info.arxiv.org/help/license/index.html). The license for a posted version is irrevocable; no license choice has been inferred from the instruction to prepare a preprint.

## Rebuilding the local records

`paper/submission_details.json` stores the supplied ORCID and selected route. Update it only with actual author choices. Run `scripts/prepare_submission.py` in the document environment with `pypdf`, then `python scripts/verify_publication.py`. The former regenerates metadata and technical PDF checks; the latter verifies saved score arithmetic and document consistency. Neither uploads a paper.
