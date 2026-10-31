"""File-type / folder filters and Explorer-style name patterns (*.jpg, report*, /regex/)."""
import os

import pytest

from conftest import fake_vec

S = os.sep


def doc(path, content="", vector=None, ftype=None):
    name = path.split("/")[-1]
    return {"file_path": path.replace("/", S), "file_name": name,
            "file_type": ftype or (name.rsplit(".", 1)[-1].lower() if "." in name else ""),
            "content": content, "vector": vector, "modified_time": 1.0}


@pytest.fixture()
def lib(db):
    db.upsert(doc("/pics/holiday.jpg", "", vector=fake_vec("beach holiday").tolist()))
    db.upsert(doc("/pics/Sunset.JPEG", "", vector=fake_vec("sunset").tolist()))
    db.upsert(doc("/pics/logo.png", "", vector=fake_vec("logo").tolist()))
    db.upsert(doc("/pics/notjpg.png", ""))
    db.upsert(doc("/pics/myjpg", "", ftype=""))                       # name ends with "jpg", no extension
    db.upsert(doc("/work/invoice_2024.pdf", "invoice total amount due 4500"))
    db.upsert(doc("/work/invoice_notes.txt", "invoice follow up call"))
    db.upsert(doc("/work/report1.docx", "quarterly invoice summary"))
    db.upsert(doc("/work/report22.docx", "annual summary"))
    db.upsert(doc("/work/sub/budget.xlsx", "invoice budget lines"))
    db.upsert(doc("/work2/invoice_x.pdf", "invoice from another folder"))
    db.upsert(doc("/misc/data.weird", "invoice in a strange format"))
    return db


def names(rs):
    return sorted(r.file_name for r in rs)


# ---------------------------------------------------------------- parsing
def test_parse_query(db):
    p = db.parse_query
    assert p("invoice").text == "invoice" and not p("invoice").has_name_rules
    assert p("*jpg").globs == ("*jpg",)
    assert p("*.jpg").exts == ("jpg",) and p("*.jpg").text == ""
    assert p(".pdf").exts == ("pdf",)
    assert p("type:pdf invoice").exts == ("pdf",) and p("type:pdf invoice").text == "invoice"
    assert p("invoice *.pdf").text == "invoice"
    assert p("report?.docx").globs == ("report?.docx",)
    # a question mark at the end of a sentence is not a wildcard
    q = p("what is the invoice total?")
    assert q.text == "what is the invoice total?" and not q.globs
    assert p("/^IMG_\\d+/").regex == "^IMG_\\d+"
    with pytest.raises(db.QuerySyntaxError):
        p("/[unclosed/")


# ---------------------------------------------------------------- wildcards, like File Explorer
def test_star_jpg_returns_only_names_ending_in_jpg(lib):
    r = lib.search("*jpg")
    assert names(r) == ["holiday.jpg", "myjpg"]          # not Sunset.JPEG, not notjpg.png
    assert all(x.match_type == "filename" for x in r)


def test_star_dot_ext_is_case_insensitive_and_covers_jpeg(lib):
    assert names(lib.search("*.jpg")) == ["Sunset.JPEG", "holiday.jpg"]
    assert names(lib.search("*.PNG")) == ["logo.png", "notjpg.png"]
    assert names(lib.search(".pdf")) == ["invoice_2024.pdf", "invoice_x.pdf"]
    assert names(lib.search("*.jpg *.png")) == ["Sunset.JPEG", "holiday.jpg", "logo.png", "notjpg.png"]
    assert names(lib.search("type:pdf")) == ["invoice_2024.pdf", "invoice_x.pdf"]


def test_prefix_suffix_and_question_mark_patterns(lib):
    assert names(lib.search("report*")) == ["report1.docx", "report22.docx"]
    assert names(lib.search("*budget*")) == ["budget.xlsx"]
    assert names(lib.search("report?.docx")) == ["report1.docx"]
    assert names(lib.search("report?")) == ["report1.docx"]       # extension may be left out
    assert names(lib.search("invoice_*.txt")) == ["invoice_notes.txt"]


def test_words_plus_pattern(lib):
    assert names(lib.search("invoice *.pdf")) == ["invoice_2024.pdf", "invoice_x.pdf"]
    assert names(lib.search("invoice report*")) == ["report1.docx"]     # file that says "invoice" AND is named report*
    assert lib.search("invoice *.pdf")[0].match_type == "keyword"
    assert lib.search("nothingmatches *.pdf") == []


def test_regex(lib):
    assert names(lib.search("/^report\\d+\\.docx$/")) == ["report1.docx", "report22.docx"]
    assert names(lib.search("/\\.(jpe?g)$/")) == ["Sunset.JPEG", "holiday.jpg"]


# ---------------------------------------------------------------- filters
def test_type_filter_limits_text_search(lib):
    assert names(lib.search("invoice")) == sorted(["invoice_2024.pdf", "invoice_notes.txt", "report1.docx", "budget.xlsx",
                                                    "invoice_x.pdf", "data.weird"])
    assert names(lib.search("invoice", types=["pdf"])) == ["invoice_2024.pdf", "invoice_x.pdf"]
    assert names(lib.search("invoice", types=["docs", "sheets"])) == ["budget.xlsx", "invoice_notes.txt", "report1.docx"]
    assert names(lib.search("invoice", types=["other"])) == ["data.weird"]          # no known family
    assert names(lib.search("invoice", types=["pdf", "other"])) == ["data.weird", "invoice_2024.pdf", "invoice_x.pdf"]


def test_type_filter_applies_to_meaning_search_too(lib):
    hits = lib.search("beach holiday")
    assert hits and hits[0].file_name == "holiday.jpg" and hits[0].match_type in ("semantic", "hybrid")
    assert lib.search("beach holiday", types=["pdf"]) == []                       # images are excluded
    assert names(lib.search("beach holiday", types=["images"])) == ["holiday.jpg"]


def test_folder_filter(lib):
    work = os.path.join(S, "work")
    assert names(lib.search("invoice", folders=[work])) == ["budget.xlsx", "invoice_2024.pdf", "invoice_notes.txt", "report1.docx"]
    # "/work" must not leak into "/work2"
    assert "invoice_x.pdf" not in names(lib.search("invoice", folders=[work]))
    assert names(lib.search("invoice", folders=[os.path.join(S, "work2")])) == ["invoice_x.pdf"]
    assert names(lib.search("invoice", folders=[work + S + "sub"])) == ["budget.xlsx"]
    assert names(lib.search("invoice", folders=[work, os.path.join(S, "misc")])) == \
        ["budget.xlsx", "data.weird", "invoice_2024.pdf", "invoice_notes.txt", "report1.docx"]


def test_folder_and_type_filters_combine_with_patterns(lib):
    work = os.path.join(S, "work")
    assert names(lib.search("*.pdf", folders=[work])) == ["invoice_2024.pdf"]
    assert names(lib.search("invoice", types=["pdf"], folders=[work])) == ["invoice_2024.pdf"]
    # ticked type AND typed type must both hold -> nothing
    assert lib.search("*.pdf", types=["images"]) == []
    assert names(lib.search("*.jpg", types=["images"], folders=[os.path.join(S, "pics")])) == ["Sunset.JPEG", "holiday.jpg"]


def test_filter_with_quote_in_folder_name(db):
    db.upsert(doc("/it's here/a.txt", "needle"))
    db.upsert(doc("/other/b.txt", "needle"))
    assert names(db.search("needle", folders=[os.path.join(S, "it's here")])) == ["a.txt"]


def test_empty_selection_means_everything(lib):
    assert len(lib.search("invoice", types=[], folders=[])) == 6
