"""Curated seed lists for the three web sources.

Edit these lists to widen/narrow the corpus. Everything downstream
(collectors, licensing table) reads from here so provenance stays in one place.
"""
from __future__ import annotations

# --------------------------------------------------------------------------- #
# Wikipedia — seed categories, recursed to WIKI_MAX_DEPTH (see collect_wikipedia)
# License: CC-BY-SA-4.0 (attribution handled via the `url` field per doc).
# --------------------------------------------------------------------------- #
WIKI_SEED_CATEGORIES = [
    "Category:Waste",
    "Category:Waste management",
    "Category:Recycling",
    "Category:Hazardous waste",
    "Category:Municipal solid waste",
    "Category:Industrial waste",
    "Category:Electronic waste",
    "Category:Landfill",
    "Category:Incineration",
    "Category:Circular economy",
    "Category:Pollution",
    "Category:Environmental crime",
    "Category:Waste legislation of the European Union",
]

# --------------------------------------------------------------------------- #
# EUR-Lex — CELEX ids of EU waste law. Fetched as consolidated English HTML.
# License: EUR-Lex reuse authorised under Commission Decision 2011/833/EU
#          (source acknowledgement required; carried in the `url`/`license` fields).
# --------------------------------------------------------------------------- #
EURLEX_CELEX = [
    # (celex, human-readable title)
    ("32008L0098", "Waste Framework Directive 2008/98/EC"),
    ("32014D0955", "Commission Decision 2014/955/EU — List of Waste"),
    ("32000D0532", "Commission Decision 2000/532/EC — establishing List of Waste"),
    ("31999L0031", "Landfill Directive 1999/31/EC"),
    ("32012L0019", "WEEE Directive 2012/19/EU"),
    ("32006L0066", "Batteries Directive 2006/66/EC"),
    ("32006R1013", "Waste Shipment Regulation (EC) 1013/2006"),
    ("32024R1157", "Waste Shipment Regulation (EU) 2024/1157"),
    ("31994L0062", "Packaging and Packaging Waste Directive 94/62/EC"),
    ("32000L0053", "End-of-Life Vehicles Directive 2000/53/EC"),
    ("32008L0099", "Directive 2008/99/EC — protection of the environment via criminal law"),
    ("32024L1203", "Directive (EU) 2024/1203 — environmental crime (recast)"),
    ("32010L0075", "Industrial Emissions Directive 2010/75/EU"),
    ("31991L0689", "Hazardous Waste Directive 91/689/EEC"),
]

# --------------------------------------------------------------------------- #
# LEA / enforcement — public agency & law-enforcement reports on waste crime.
# License: VARIES. Each entry MUST be license-checked before use; default
# marker is "verify". Prefer EU/agency reports with open reuse terms.
# Provide direct file URLs (PDF or HTML). PDFs need pypdf (see tokenize env).
# --------------------------------------------------------------------------- #
LEA_SOURCES = [
    # (doc_key, url, kind{"pdf"|"html"}, license, title)
    # --- Starter set: replace/extend with URLs you have verified as public ---
    # ("europol_socta_2021_env", "https://www.europol.europa.eu/.../socta2021.pdf",
    #  "pdf", "verify", "Europol EU SOCTA 2021 — environmental crime section"),
    # ("impel_waste_2023", "https://www.impel.eu/.../report.pdf",
    #  "pdf", "verify", "IMPEL waste enforcement report 2023"),
    # ("eea_waste_2023", "https://www.eea.europa.eu/.../waste.pdf",
    #  "pdf", "verify", "EEA report on waste"),
]
