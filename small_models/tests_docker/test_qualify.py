from vm.libs.qualify import has_compiled_sources, visible_half


def test_visible_half_is_seeded_and_half():
    ids = [f"tests/t.py::test_{i}" for i in range(101)]
    a = visible_half(ids, "lib")
    assert len(a) == 50 and a == visible_half(list(reversed(ids)), "lib")
    assert a != visible_half(ids, "other")


def test_compiled_sources_are_detected(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "a.py").write_text("x = 1\n")
    assert not has_compiled_sources(tmp_path)
    (tmp_path / "pkg" / "fast.pyx").write_text("cdef int x\n")
    assert has_compiled_sources(tmp_path)
