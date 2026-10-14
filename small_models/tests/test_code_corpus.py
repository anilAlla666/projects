import io
import json
import tarfile

from vm.corpus.code import filter_files, safe_members, top_level_names


def make_tar(tmp_path, files, name="pkg-1.0.tar.gz", link=None):
    p = tmp_path / name
    with tarfile.open(p, "w:gz") as t:
        for path, text in files.items():
            data = text.encode()
            ti = tarfile.TarInfo(path)
            ti.size = len(data)
            t.addfile(ti, io.BytesIO(data))
        if link:
            ti = tarfile.TarInfo(link)
            ti.type = tarfile.SYMTYPE
            ti.linkname = "/etc/passwd"
            t.addfile(ti)
    return p


def test_extraction_refuses_traversal_links_and_non_python(tmp_path):
    p = make_tar(tmp_path, {"pkg-1.0/pkg/a.py": "x = 1\n", "../evil.py": "bad\n", "/abs.py": "bad\n",
                            "pkg-1.0/README.md": "readme"}, link="pkg-1.0/pkg/link.py")
    got = dict(safe_members(p))
    assert list(got) == ["pkg/a.py"]


def test_top_level_names():
    assert top_level_names(["six.py", "pkg/__init__.py", "pkg/sub/x.py", "tests/t.py"]) == {"six", "pkg"}


def test_filter_drops_vendored_held_out_code_and_duplicates():
    files = [("good", "good/a.py", "a = 1\n"), ("good", "good/_vendor/six/x.py", "s = 1\n"),
             ("other", "other/b.py", "a = 1\n"), ("other", "other/c.py", "c = 2\n")]
    kept = filter_files(files, held_out_imports={"six"})
    assert [(p, path) for p, path, _ in kept] == [("good", "good/a.py"), ("other", "other/c.py")]


def test_filter_has_teeth():
    """Without the held-out name, the vendored file stays: the filter is what removes it."""
    files = [("good", "good/_vendor/six/x.py", "s = 1\n")]
    assert filter_files(files, held_out_imports=set()) == files
