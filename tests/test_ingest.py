from leaflet_rag.ingest import chunk_document, split_sections


def test_sections_found_in_order(chunks):
    sections = [c.section for c in chunks if c.doc == "fictamol-250mg"]
    assert {"1", "2", "3", "4", "5", "6"} <= set(sections)


def test_table_of_contents_is_not_mistaken_for_content(chunks):
    side_effects = [c for c in chunks if c.doc == "fictamol-250mg" and c.section == "4"]
    assert any("nausea" in c.text for c in side_effects)


def test_text_before_first_section_kept_as_intro():
    sections = split_sections("Leaflet header\n1. What X is and what it is used for\nBody")
    assert sections[0][0] == "0" and sections[1][0] == "1"


def test_long_sections_are_windowed_with_overlap():
    body = "\n\n".join(" ".join(f"w{p}_{i}" for i in range(50)) for p in range(10))
    text = "4. Possible side effects\n" + body
    chunks = chunk_document("doc", text, max_words=120, overlap=20)
    assert len(chunks) > 1
    assert all(len(c.text.split()) <= 120 for c in chunks)
    first, second = chunks[0].text.split(), chunks[1].text.split()
    assert first[-20:] == second[:20]


def test_chunk_ids_unique(chunks):
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))


def test_smpc_before_leaflet_is_dropped():
    from leaflet_rag.ingest import leaflet_part

    full = (
        "ANNEX I SUMMARY OF PRODUCT CHARACTERISTICS\n4.8 Undesirable effects: hepatotoxicity\n"
        "ANNEX III B. PACKAGE LEAFLET\nPackage leaflet: Information for the user\n"
        "4. Possible side effects\nHeadache."
    )
    part = leaflet_part(full)
    assert part.startswith("Package leaflet") and "hepatotoxicity" not in part
